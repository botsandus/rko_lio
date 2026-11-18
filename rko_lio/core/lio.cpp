/*
 * MIT License
 *
 * Copyright (c) 2025 Meher V.R. Malladi.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <UTL/profiler.hpp>
#include <spdlog/spdlog.h>

#include "deskew.hpp"
#include "error.hpp"
#include "lio.hpp"
#include "preprocess_scan.hpp"
#include "util.hpp"
// other
#include <sophus/se3.hpp>
// tbb
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>
// stl
#include <algorithm>
#include <array>
#include <functional>
#include <numeric>
#include <sstream>

namespace {
using namespace rko_lio::core;

constexpr Scalar NEGLIGIBLE_EXTRINSIC = 1e-6;
constexpr auto EPSILON_TIME = std::chrono::nanoseconds(10);
constexpr auto MAX_SCAN_TO_SCAN_DELTA = std::chrono::seconds(1);
// 10% of a 10 Hz lidar's period: no lidar finishes a scan this soon after the last, so a closer end stamp is a stamp
// error, and the velocity estimate (pose delta / dt) over it would turn ICP noise into a large velocity
constexpr auto MIN_SCAN_TO_SCAN_DELTA = std::chrono::milliseconds(10);

template <typename T>
inline std::string streamed(const T& value) {
  std::ostringstream oss;
  oss << value;
  return oss.str();
}

inline void transform_points(const Sophus::SE3s& T, Vector3sVector& points) {
  std::transform(points.begin(), points.end(), points.begin(), [&](const auto& point) { return T * point; });
}

void to_base_frame(std::vector<Vector3sVector>& scan, const std::span<const Sophus::SE3s> lidar2base) {
  for (std::size_t cloud = 0; cloud < scan.size(); ++cloud) {
    if (lidar2base[cloud].log().norm() >= NEGLIGIBLE_EXTRINSIC) {
      transform_points(lidar2base[cloud], scan[cloud]);
    }
  }
}

struct LinearSystem {
  Eigen::Matrix6s H;
  Eigen::Vector6s b;
  Scalar cost;
  int correspondences;
};
LinearSystem build_icp_linear_system(const Sophus::SE3s& current_pose,
                                     const rko_lio::core::Vector3sVector& keypoints,
                                     const rko_lio::core::VoxelHashMap& voxel_map,
                                     const Scalar& max_correspondence_distance) {
  using IcpAccum = std::tuple<Eigen::Matrix6s, Eigen::Vector6s, Scalar, int>;

  auto linear_system_for_one_point = [](const Eigen::Vector3s& source, const Eigen::Vector3s& residual) {
    // world frame this is R * [I | -hat(source)]; write the residual in the body frame, the R cancels, and the compute
    // is cheaper
    Eigen::Matrix3_6s J_r;
    J_r.block<3, 3>(0, 0) = Eigen::Matrix3s::Identity();
    J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3s::hat(source);
    return IcpAccum(J_r.transpose() * J_r,      // JTJ
                    J_r.transpose() * residual, // JTr
                    residual.squaredNorm(),     // chi
                    1);                         // correspondence
  };

  const Eigen::Matrix3s rotation_transpose = current_pose.so3().inverse().matrix();

  // The only parallel part
  using points_iterator = std::vector<Eigen::Vector3s>::const_iterator;
  const auto& [H_icp, b_icp, chi_icp, correspondences_counter] = tbb::parallel_reduce(
      // Range
      tbb::blocked_range<points_iterator>{keypoints.cbegin(), keypoints.cend()},
      // Identity
      IcpAccum(Eigen::Matrix6s::Zero(), Eigen::Vector6s::Zero(), 0.0, 0),
      // 1st Lambda: Parallel computation
      [&](const tbb::blocked_range<points_iterator>& r, IcpAccum J) -> IcpAccum {
        auto& [H, b, chi, correspondences] = J;
        for (const Eigen::Vector3s& point : r) {
          const Eigen::Vector3s transformed_point = current_pose * point;
          const auto closest_neighbor = voxel_map.get_closest_neighbor(transformed_point, max_correspondence_distance);
          if (!closest_neighbor.has_value()) {
            continue;
          }
          const Eigen::Vector3s residual = rotation_transpose * (transformed_point - *closest_neighbor);
          const auto& [point_H, point_b, point_chi, point_n] = linear_system_for_one_point(point, residual);
          H += point_H;
          b += point_b;
          chi += point_chi;
          correspondences += point_n;
        }
        return J;
      },
      // 2nd Lambda: Parallel reduction of the private Jacobians
      [](IcpAccum lhs, const IcpAccum& rhs) {
        auto& [lhs_H, lhs_b, lhs_chi, lhs_n] = lhs;
        const auto& [rhs_H, rhs_b, rhs_chi, rhs_n] = rhs;
        lhs_H += rhs_H;
        lhs_b += rhs_b;
        lhs_chi += rhs_chi;
        lhs_n += rhs_n;
        return lhs;
      });

  if (correspondences_counter == 0) {
    throw RegistrationError("ICP found no correspondences");
  }

  return {
      .H = H_icp / correspondences_counter,
      .b = b_icp / correspondences_counter,
      .cost = static_cast<Scalar>(0.5 * chi_icp / correspondences_counter),
      .correspondences = correspondences_counter,
  };
}

struct IcpResult {
  Sophus::SE3s pose;
  std::size_t iterations = 0;
  int correspondences = 0; // of the last iteration
  Scalar cost = 0;         // of the last iteration
};

IcpResult icp(const Vector3sVector& keypoints,
              const VoxelHashMap& voxel_map,
              const Sophus::SE3s& initial_guess,
              const LIO::Config& config) {
  IcpResult result{.pose = initial_guess};
  Sophus::SE3s& current_pose = result.pose;
  Scalar previous_chi = -1;

  for (size_t i = 0; i < config.max_iterations; ++i) {
    const auto& [H, b, chi, correspondences] =
        build_icp_linear_system(current_pose, keypoints, voxel_map, config.max_correspondence_distance);
    result.iterations = i + 1;
    result.correspondences = correspondences;
    result.cost = chi;

    const Eigen::Vector6s dx = std::invoke([&]() -> Eigen::Vector6s {
      if (!config.flat) {
        return H.ldlt().solve(-b);
      }
      constexpr std::array<int, 3> x_y_yaw{0, 1, 5};
      Eigen::Vector6s flat_dx = Eigen::Vector6s::Zero();
      flat_dx(x_y_yaw) = H(x_y_yaw, x_y_yaw).ldlt().solve(-b(x_y_yaw)).eval();
      return flat_dx;
    });
    current_pose = current_pose * Sophus::SE3s::exp(dx);

    // Ceres function_tolerance style convergence check
    const Scalar cost_change = std::abs(previous_chi - chi);
    const bool converged = previous_chi > 0 && cost_change <= config.convergence_criterion * previous_chi;
    previous_chi = chi;
    if (converged || i == (config.max_iterations - 1)) {
      break;
    }
  }
  return result;
}

inline Sophus::SO3s align_accel_to_z_world(const Eigen::Vector3s& accel) {
  //  unobservable in the gravity direction, and the z in R.log() will always be 0
  const Eigen::Vector3s z_world = {0.0, 0.0, 1.0};
  const Eigen::Quaternions quat_accel = Eigen::Quaternions::FromTwoVectors(accel, z_world);
  return Sophus::SO3s(quat_accel);
}
} // namespace

// ==========================
//   actual LIO class stuff
// ==========================

namespace rko_lio::core {

LIO::LIO(const Config& config_)
    : config(config_),
      map(config_.voxel_size, config_.max_range),
      arena(config_.max_num_threads > 0 ? config_.max_num_threads : tbb::task_arena::automatic) {}

// ==========================
//          private
// ==========================

void LIO::initialize(const Nsec lidar_time) {
  if (interval_stats.imu_count == 0) {
    spdlog::warn("Cannot initialize. No imu measurements received.");
    initialized_ = true;
    return;
  }

  const Eigen::Vector3s avg_accel = interval_stats.imu_acceleration_sum / interval_stats.imu_count;
  const Eigen::Vector3s avg_gyro = interval_stats.angular_velocity_sum / interval_stats.imu_count;

  const Sophus::SO3s initial_rotation = align_accel_to_z_world(avg_accel);
  if (!config.flat) {
    lidar_state.pose.so3() = initial_rotation;
  }
  imu_state = lidar_state;

  lidar_state.time = lidar_time;
  imu_state.time = lidar_time;

  const Eigen::Vector3s local_gravity = imu_state.pose.so3().inverse() * gravity();
  imu_bias.accelerometer = avg_accel + local_gravity;
  imu_bias.gyroscope = avg_gyro;

  initialized_ = true;
  spdlog::info("Odometry map frame initialized using {} IMU measurements. Estimated initial rotation [se(3)] is {}",
               interval_stats.imu_count, streamed(imu_state.pose.so3().log().transpose()));
  spdlog::info("Estimated accel bias: {}, gyro bias: {}", streamed(imu_bias.accelerometer.transpose()),
               streamed(imu_bias.gyroscope.transpose()));
}

std::vector<Vector3sVector> LIO::bootstrap_first_scan(std::vector<Vector3sVector> scan,
                                                      const std::span<const Sophus::SE3s> lidar2base,
                                                      const Nsec current_lidar_time) {
  lidar_state.time = current_lidar_time;
  imu_state = lidar_state;
  to_base_frame(scan, lidar2base);
  auto preproc = preprocess_scan(std::move(scan), config);
  if (!config.initialization_phase) {
    map.update(config.double_downsample ? preproc.map_points : preproc.keypoints, lidar_state.pose);
    spdlog::info("Odometry map frame initialized with first lidar scan.");
  }
  return std::move(preproc.filtered_scan);
}

MotionPrior LIO::motion_prior_from_imu(const Nsec current_lidar_time) {
  const Nsec start = lidar_state.time;
  const Eigen::Vector3s& linear_velocity = lidar_state.linear_velocity;
  if (config.initialization_phase && !initialized_) {
    // assume static and
    initialize(current_lidar_time);
    return {.start_time = start, .linear_velocity = linear_velocity};
  }
  if (interval_stats.imu_count == 0) {
    spdlog::warn("No Imu measurements in interval to average. Assuming constant velocity motion.");
    return {.start_time = start, .linear_velocity = linear_velocity, .angular_velocity = lidar_state.angular_velocity};
  }
  const Eigen::Vector3s avg_body_accel = interval_stats.body_acceleration_sum / interval_stats.imu_count;
  const Eigen::Vector3s avg_ang_vel = interval_stats.angular_velocity_sum / interval_stats.imu_count;
  if (avg_body_accel.norm() > 50.0) {
    spdlog::warn("Erratic body acceleration computed, norm > 50 m/s2. Either IMU data is corrupted, or you should "
                 "report an issue.");
  }
  return {
      .start_time = start,
      .linear_velocity = linear_velocity,
      .acceleration = avg_body_accel,
      .angular_velocity = avg_ang_vel,
  };
}

// ==========================
//          public
// ==========================

// ============================ imu ===============================

void LIO::add_imu_measurement(const ImuControl& base_imu) {
  if (lidar_state.time < EPSILON_TIME) {
    static bool warning_skip_till_first_lidar = false;
    if (!warning_skip_till_first_lidar) {
      spdlog::warn("Skipping IMU, waiting for first LiDAR message.");
      warning_skip_till_first_lidar = true;
    }
    last_real_imu_time_ = base_imu.time;
    last_real_base_imu_ang_vel_ = base_imu.angular_velocity;
    return;
  }

  if (imu_state.time < EPSILON_TIME) {
    imu_state = lidar_state;
  }

  const auto dt = to_seconds<Scalar>(base_imu.time - imu_state.time);

  if (dt < 0.0) {
    spdlog::warn("Skipping IMU message from the past.");
    return;
  }

  const Eigen::Vector3s unbiased_ang_vel = std::invoke([&]() -> Eigen::Vector3s {
    const Eigen::Vector3s unbiased = base_imu.angular_velocity - imu_bias.gyroscope;
    return config.flat ? Eigen::Vector3s(0, 0, unbiased.z()) : unbiased;
  });
  const Eigen::Vector3s unbiased_accel = base_imu.acceleration - imu_bias.accelerometer;

  const Eigen::Vector3s local_gravity = imu_state.pose.so3().inverse() * gravity();
  const Eigen::Vector3s compensated_accel = std::invoke([&]() -> Eigen::Vector3s {
    const Eigen::Vector3s compensated = unbiased_accel + local_gravity;
    return config.flat ? Eigen::Vector3s(compensated.x(), compensated.y(), 0) : compensated;
  });

  // imu state update
  Eigen::Vector6s tau;
  tau.head<3>() = imu_state.linear_velocity * dt + compensated_accel * square(dt) / 2;
  tau.tail<3>() = unbiased_ang_vel * dt;
  imu_state.pose = imu_state.pose * Sophus::SE3s::exp(tau);
  imu_state.linear_velocity += compensated_accel * dt;
  imu_state.angular_velocity = unbiased_ang_vel;
  imu_state.time = base_imu.time;

  interval_stats.update(unbiased_ang_vel, unbiased_accel, compensated_accel);

  last_real_imu_time_ = base_imu.time;
  last_real_base_imu_ang_vel_ = base_imu.angular_velocity;
}

void LIO::add_imu_measurement(const Sophus::SE3s& extrinsic_imu2base, const ImuControl& raw_imu) {
  if (extrinsic_imu2base.log().norm() < NEGLIGIBLE_EXTRINSIC) {
    add_imu_measurement(raw_imu);
    return;
  }

  if (last_real_imu_time_ < EPSILON_TIME) {
    // skip IMU message as we need a previous imu time for extrinsic compensation
    last_real_imu_time_ = raw_imu.time;
    return;
  }

  // accounting for the transport-rate
  ImuControl base_imu = raw_imu;
  const Sophus::SO3s& extrinsic_rotation = extrinsic_imu2base.so3();
  base_imu.angular_velocity = extrinsic_rotation * raw_imu.angular_velocity;

  const Eigen::Vector3s& lever_arm = -1 * extrinsic_imu2base.translation();
  const Nsec dt = raw_imu.time - last_real_imu_time_;

  const Eigen::Vector3s angular_acceleration = std::invoke([&]() -> Eigen::Vector3s {
    if (std::chrono::abs(dt) < std::chrono::microseconds(200)) {
      // if dt is less than the equivalent of a 5000 Hz imu, assuming zero ang accel,
      // causes numerical issues otherwise
      static bool warning_imu_too_close = false;
      if (!warning_imu_too_close) {
        spdlog::warn("Received IMU message with a very short delta to previous IMU message. Ignoring all such "
                     "messages.");
        warning_imu_too_close = true;
      }
      return Eigen::Vector3s::Zero();
    } else {
      const Eigen::Vector3s angular_acceleration =
          (base_imu.angular_velocity - last_real_base_imu_ang_vel_) / to_seconds(dt);
      return angular_acceleration;
    }
  });

  base_imu.acceleration = extrinsic_rotation * raw_imu.acceleration + angular_acceleration.cross(lever_arm) +
                          base_imu.angular_velocity.cross(base_imu.angular_velocity.cross(lever_arm));

  this->add_imu_measurement(base_imu);
}

// ============================ lidar ===============================

std::vector<Vector3sVector> LIO::register_payload(std::vector<Vector3sVector> scan,
                                                  const std::span<const Timestamps> timestamps,
                                                  const std::span<const Sophus::SE3s> lidar2base) {
  const std::size_t point_count = std::transform_reduce(scan.cbegin(), scan.cend(), std::size_t{0}, std::plus<>(),
                                                        [](const Vector3sVector& cloud) { return cloud.size(); });
  scan_stats = {.input_points = point_count, .imu_samples = interval_stats.imu_count};
  if (scan.empty() || scan.size() != timestamps.size() || scan.size() != lidar2base.size()) {
    throw InputError("LiDAR payload of " + std::to_string(scan.size()) + " clouds has " +
                     std::to_string(timestamps.size()) + " timestamp entries and " + std::to_string(lidar2base.size()) +
                     " extrinsics; it needs one of each per cloud.");
  }
  for (std::size_t cloud = 0; cloud < scan.size(); ++cloud) {
    if (scan[cloud].size() != timestamps[cloud].per_point.size()) {
      throw InputError("LiDAR timestamps size (" + std::to_string(timestamps[cloud].per_point.size()) +
                       ") does not match scan (" + std::to_string(scan[cloud].size()) + ").");
    }
  }
  const Nsec scan_start = std::ranges::min(timestamps, {}, &Timestamps::min).min;
  const Nsec current_lidar_time = std::ranges::max(timestamps, {}, &Timestamps::max).max;
  if (lidar_state.time >= EPSILON_TIME && current_lidar_time - lidar_state.time < MIN_SCAN_TO_SCAN_DELTA) {
    spdlog::warn("Dropping a LiDAR scan that ends {} ms after the previous one.",
                 std::chrono::duration<double, std::milli>(current_lidar_time - lidar_state.time).count());
    return {};
  }
  if (fragment_check_.is_fragment(to_seconds<Scalar>(current_lidar_time - scan_start),
                                  static_cast<Scalar>(point_count))) {
    spdlog::warn("Dropping a LiDAR scan fragment of {} points.", point_count);
    return {};
  }

  if (lidar_state.time < EPSILON_TIME) {
    return bootstrap_first_scan(std::move(scan), lidar2base, current_lidar_time);
  }

  if (std::chrono::abs(current_lidar_time - lidar_state.time) > MAX_SCAN_TO_SCAN_DELTA) {
    // the imu prior below extrapolates across the gap, and falls back to constant velocity if the imu dropped out too
    spdlog::warn("{} seconds delta to the previous scan. Extrapolating the motion prior across it.",
                 to_seconds(current_lidar_time - lidar_state.time));
  }

  const MotionPrior motion = motion_prior_from_imu(current_lidar_time);

  const Sophus::SE3s initial_guess = lidar_state.pose * relative_pose_at_time(motion, current_lidar_time);

  if (config.deskew) {
    UTL_PROFILER_SCOPE("rko_lio: deskew");
    const auto gap_to_scan_start = to_seconds<Scalar>(scan_start - motion.start_time);
    const MotionPrior scan_motion{
        .start_time = scan_start,
        .linear_velocity = motion.linear_velocity + motion.acceleration * gap_to_scan_start,
        .acceleration = motion.acceleration,
        .angular_velocity = motion.angular_velocity,
    };
    for (std::size_t cloud = 0; cloud < scan.size(); ++cloud) {
      deskew_scan(scan[cloud], timestamps[cloud], current_lidar_time, scan_motion, lidar2base[cloud]);
    }
  } else {
    to_base_frame(scan, lidar2base);
  }
  auto preproc_result = std::invoke([&] {
    UTL_PROFILER_SCOPE("rko_lio: preprocess");
    return preprocess_scan(std::move(scan), config);
  });
  scan_stats.keypoints = preproc_result.keypoints.size();

  if (preproc_result.keypoints.size() < 10) {
    const std::string error_msg = "Too few ICP keypoints: " + std::to_string(preproc_result.keypoints.size()) +
                                  " (input scan size: " + std::to_string(point_count) +
                                  ", voxel_size: " + std::to_string(config.voxel_size) +
                                  "). Scan empty/corrupt, or downsampling is too aggressive.";
    throw InputError(error_msg);
  }

  const Vector3sVector& map_input = config.double_downsample ? preproc_result.map_points : preproc_result.keypoints;

  if (!map.empty()) {
    UTL_PROFILER_SCOPE("rko_lio: ICP");
    const IcpResult icp_result =
        arena.execute([&] { return icp(preproc_result.keypoints, map, initial_guess, config); });
    const Sophus::SE3s& optimized_pose = icp_result.pose;
    scan_stats.icp_iterations = icp_result.iterations;
    scan_stats.correspondences = icp_result.correspondences;
    scan_stats.icp_cost = icp_result.cost;

    // estimate velocities and accelerations from the new pose
    const auto dt = to_seconds<Scalar>(current_lidar_time - lidar_state.time);
    const Sophus::SE3s pose_delta = lidar_state.pose.inverse() * optimized_pose;
    const Eigen::Vector6s local_velocity = pose_delta.log() / dt;
    const Eigen::Vector3s local_linear_acceleration =
        (local_velocity.head<3>() - pose_delta.so3().inverse() * lidar_state.linear_velocity) / dt;

    // update
    lidar_state.pose = optimized_pose;
    lidar_state.linear_velocity = local_velocity.head<3>();
    lidar_state.angular_velocity = local_velocity.tail<3>();
    lidar_state.linear_acceleration = local_linear_acceleration;
  }
  // even if map is empty, time should still update
  lidar_state.time = current_lidar_time;
  // reset imu to last know lidar state
  imu_state = lidar_state;
  // reset imu averages
  interval_stats.reset();

  {
    UTL_PROFILER_SCOPE("rko_lio: map update");
    map.update(map_input, lidar_state.pose);
  }

  return std::move(preproc_result.filtered_scan);
}

void LIO::restart() {
  ++reset_count;
  config.initialization_phase = false;
  map.clear();
  lidar_state = {};
  imu_state = {};
  interval_stats.reset();
  initialized_ = false;
}
} // namespace rko_lio::core

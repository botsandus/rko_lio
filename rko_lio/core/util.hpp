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

#pragma once
#include "rko_lio/core/scalar.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <numeric>
#include <sophus/se3.hpp>
#include <span>
#include <utility>
#include <vector>

// Scalar is either float or double, as configured
namespace Eigen {
using Vector3s = Vector3<rko_lio::core::Scalar>;
using Vector4s = Vector4<rko_lio::core::Scalar>;
using Matrix3s = Matrix3<rko_lio::core::Scalar>;
using Matrix4s = Matrix4<rko_lio::core::Scalar>;
using Matrix3_6s = Matrix<rko_lio::core::Scalar, 3, 6>;
using Vector6s = Matrix<rko_lio::core::Scalar, 6, 1>;
using Matrix6s = Matrix<rko_lio::core::Scalar, 6, 6>;
using Quaternions = Quaternion<rko_lio::core::Scalar>;
using Vector3i8 = Vector3<std::int8_t>;
} // namespace Eigen

namespace Sophus {
using SE3s = SE3<rko_lio::core::Scalar>;
using SO3s = SO3<rko_lio::core::Scalar>;
} // namespace Sophus

namespace rko_lio::core {
using Vector3sVector = std::vector<Eigen::Vector3s>;

using Nsec = std::chrono::nanoseconds;
using TimestampVector = std::vector<Nsec>;

// constants and util funcs
constexpr Scalar square(Scalar x) { return x * x; }
constexpr Scalar GRAVITY_MAG = 9.8107;
inline Eigen::Vector3s gravity() { return {0, 0, -GRAVITY_MAG}; }

// always divides in double, then narrows, so an absolute timestamp does not lose its magnitude
template <typename T = double>
constexpr T to_seconds(const Nsec d) {
  return static_cast<T>(std::chrono::duration<double>(d).count());
}

// data structs
struct Timestamps {
  Nsec min;
  Nsec max;
  TimestampVector per_point;
};

inline std::size_t point_count(const std::span<const Vector3sVector> clouds) {
  return std::transform_reduce(clouds.begin(), clouds.end(), std::size_t{0}, std::plus<>(),
                               [](const Vector3sVector& cloud) { return cloud.size(); });
}

// the first and the last point stamp over a payload's clouds
inline std::pair<Nsec, Nsec> payload_start_end(const std::span<const Timestamps> timestamps) {
  return {std::ranges::min(timestamps, {}, &Timestamps::min).min,
          std::ranges::max(timestamps, {}, &Timestamps::max).max};
}

struct State {
  Nsec time{0};
  Sophus::SE3s pose;
  Eigen::Vector3s linear_velocity = Eigen::Vector3s::Zero();
  Eigen::Vector3s angular_velocity = Eigen::Vector3s::Zero();
  Eigen::Vector3s linear_acceleration = Eigen::Vector3s::Zero();
};

/** Constant-motion model anchored at the last registered state */
struct MotionPrior {
  Nsec start_time{0};
  Eigen::Vector3s linear_velocity = Eigen::Vector3s::Zero();
  Eigen::Vector3s acceleration = Eigen::Vector3s::Zero();
  Eigen::Vector3s angular_velocity = Eigen::Vector3s::Zero();
};

/** Pose change from start_time to time */
inline Sophus::SE3s relative_pose_at_time(const MotionPrior& prior, const Nsec time) {
  const auto dt = to_seconds<Scalar>(time - prior.start_time);
  Eigen::Vector6s tau;
  tau.head<3>() = prior.linear_velocity * dt + (prior.acceleration * square(dt) / 2);
  tau.tail<3>() = prior.angular_velocity * dt;
  return Sophus::SE3s::exp(tau);
}

struct ImuBias {
  Eigen::Vector3s accelerometer = Eigen::Vector3s::Zero();
  Eigen::Vector3s gyroscope = Eigen::Vector3s::Zero();
};

struct ImuControl {
  Nsec time{0};
  Eigen::Vector3s acceleration = Eigen::Vector3s::Zero();
  Eigen::Vector3s angular_velocity = Eigen::Vector3s::Zero();
};

/** Tells a driver's scan fragment (a frame closed early, its tail published on its own) from a full scan, by running
 *  averages over the full ones. It is fed a lidar's scans (in the merge) or the registered payloads (in LIO). */
class ScanFragmentCheck {
public:
  /** True when both the duration and the point count are below a fixed fraction of the running averages; a full one
   *  updates them. */
  bool is_fragment(const Scalar duration_s, const Scalar point_count) {
    constexpr Scalar SCAN_FRAGMENT_FRACTION = 0.25;
    constexpr Scalar LOWPASS_TIME_CONSTANT = 10;
    if (duration_s < SCAN_FRAGMENT_FRACTION * lowpass_duration_s_ &&
        point_count < SCAN_FRAGMENT_FRACTION * lowpass_point_count_) {
      return true;
    }
    const Scalar alpha = lowpass_point_count_ > 0 ? 1 / LOWPASS_TIME_CONSTANT : 1;
    lowpass_duration_s_ += alpha * (duration_s - lowpass_duration_s_);
    lowpass_point_count_ += alpha * (point_count - lowpass_point_count_);
    return false;
  }

private:
  Scalar lowpass_duration_s_ = 0;
  Scalar lowpass_point_count_ = 0;
};

/** One payload's statistics: its points and IMU samples, the ICP's input and last iteration. */
struct PayloadStats {
  std::size_t input_points = 0; // points in the payload as given
  std::size_t keypoints = 0;    // fed to the ICP
  int imu_samples = 0;          // since the previous payload
  std::size_t icp_iterations = 0;
  int correspondences = 0; // of the last ICP iteration
  Scalar icp_cost = 0;     // half the mean squared residual of the last ICP iteration
};

/** Accumulated IMU statistics over the interval between consecutive payloads. */
struct IntervalStats {
  /** Number of IMU samples accumulated during this interval. */
  int imu_count = 0;

  /** Sum of unbiased angular velocity samples. */
  Eigen::Vector3s angular_velocity_sum = Eigen::Vector3s::Zero();

  /** Sum of gravity-compensated body-frame accelerations. */
  Eigen::Vector3s body_acceleration_sum = Eigen::Vector3s::Zero();

  /** Sum of unbiased raw IMU accelerations. */
  Eigen::Vector3s imu_acceleration_sum = Eigen::Vector3s::Zero();

  /**
   * Update accumulated statistics with a new IMU measurement.
   * @param unbiased_ang_vel Unbiased angular velocity.
   * @param uncompensated_unbiased_accel Uncompensated, unbiased acceleration.
   * @param compensated_accel Gravity-compensated acceleration.
   */
  void update(const Eigen::Vector3s& unbiased_ang_vel,
              const Eigen::Vector3s& uncompensated_unbiased_accel,
              const Eigen::Vector3s& compensated_accel) {
    ++imu_count;
    angular_velocity_sum += unbiased_ang_vel;
    imu_acceleration_sum += uncompensated_unbiased_accel;

    body_acceleration_sum += compensated_accel;
  }

  /** Reset all accumulated statistics to zero. */
  void reset() {
    imu_count = 0;
    angular_velocity_sum.setZero();
    body_acceleration_sum.setZero();
    imu_acceleration_sum.setZero();
  }
};

}; // namespace rko_lio::core

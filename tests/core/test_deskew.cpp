#include "rko_lio/core/deskew.hpp"
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

using rko_lio::core::deskew_scan;
using rko_lio::core::MotionPrior;
using rko_lio::core::Nsec;
using rko_lio::core::relative_pose_at_time;
using rko_lio::core::Scalar;
using rko_lio::core::Timestamps;
using rko_lio::core::TimestampVector;
using rko_lio::core::to_seconds;
using rko_lio::core::Vector3sVector;

namespace {
constexpr Nsec SCAN_START{1'794'831'720'950'227'976};

// Reference: the SE(3) exp about a fixed axis in closed form (Rodrigues, left Jacobian times rho), per point.
Vector3sVector deskew_per_point(const Vector3sVector& scan, const Timestamps& timestamps, const MotionPrior& motion) {
  const Sophus::SE3s start2reference = relative_pose_at_time(motion, timestamps.max).inverse();
  const Eigen::Matrix3s& start2reference_rotation = start2reference.so3().matrix();
  const Eigen::Vector3s& start2reference_translation = start2reference.translation();
  const Scalar angular_speed = std::max(motion.angular_velocity.norm(), static_cast<Scalar>(1e-6));
  const Scalar inverse_angular_speed = 1 / angular_speed;
  const Eigen::Vector3s axis = motion.angular_velocity * inverse_angular_speed;
  Vector3sVector deskewed(scan.size());
  std::transform(scan.cbegin(), scan.cend(), timestamps.per_point.cbegin(), deskewed.begin(),
                 [&](const Eigen::Vector3s& point, const Nsec timestamp) {
                   const auto dt = to_seconds<Scalar>(timestamp - motion.start_time);
                   const Scalar theta = angular_speed * dt;
                   const Scalar sin_theta = std::sin(theta);
                   const Scalar one_minus_cos_theta = 1 - std::cos(theta);
                   const Eigen::Vector3s mean_linear_velocity = motion.linear_velocity + motion.acceleration * dt / 2;
                   const Eigen::Vector3s K_times_p = axis.cross(point);
                   const Eigen::Vector3s K_times_linear_velocity = axis.cross(mean_linear_velocity);
                   const Eigen::Vector3s R_times_p =
                       point + sin_theta * K_times_p + one_minus_cos_theta * axis.cross(K_times_p);
                   const Eigen::Vector3s V_times_rho =
                       dt * mean_linear_velocity +
                       (one_minus_cos_theta * inverse_angular_speed) * K_times_linear_velocity +
                       ((theta - sin_theta) * inverse_angular_speed) * axis.cross(K_times_linear_velocity);
                   const Eigen::Vector3s point_at_start = R_times_p + V_times_rho;
                   return Eigen::Vector3s(start2reference_rotation * point_at_start + start2reference_translation);
                 });
  return deskewed;
}

Vector3sVector random_points(const std::size_t count) {
  std::mt19937 generator(42);
  std::uniform_real_distribution<Scalar> coordinate(-60, 60);
  Vector3sVector points(count);
  std::generate(points.begin(), points.end(),
                [&] { return Eigen::Vector3s(coordinate(generator), coordinate(generator), coordinate(generator)); });
  return points;
}

// Row-major organised scan: each of `rows` channels fires once per column, all channels of a column at the same
// stamp, so equal stamps are `columns` points apart.
Timestamps organised_timestamps(const int rows, const int columns, const Nsec duration) {
  TimestampVector per_point;
  per_point.reserve(static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns));
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      per_point.push_back(SCAN_START + duration * column / (columns - 1));
    }
  }
  return {.min = SCAN_START, .max = SCAN_START + duration, .per_point = std::move(per_point)};
}

Scalar max_relative_difference(const Vector3sVector& scan,
                               const Timestamps& timestamps,
                               const MotionPrior& motion,
                               const Sophus::SE3s& lidar2base,
                               const bool per_stamp) {
  Vector3sVector base_scan(scan.size());
  std::transform(scan.cbegin(), scan.cend(), base_scan.begin(),
                 [&](const Eigen::Vector3s& point) { return Eigen::Vector3s(lidar2base * point); });
  const Vector3sVector expected = deskew_per_point(base_scan, timestamps, motion);
  Vector3sVector deskewed = scan;
  deskew_scan(deskewed, timestamps, timestamps.max, motion, lidar2base, per_stamp);
  Scalar max_difference = 0;
  for (std::size_t i = 0; i < scan.size(); ++i) {
    max_difference = std::max(max_difference, (deskewed[i] - expected[i]).norm() / std::max(scan[i].norm(), Scalar{1}));
  }
  return max_difference;
}

// Both forms are about ten float operations deep on values bounded by the point's range plus a translation under a
// metre, so each is within ~5 eps of the exact value relative to max(|p|, 1 m); 32 eps leaves a margin of three.
constexpr Scalar TOLERANCE = 32 * std::numeric_limits<Scalar>::epsilon();

// a lidar mounted off the base origin and yawed by about 130 degrees
Sophus::SE3s mount() {
  return {Sophus::SO3s::exp(Eigen::Vector3s(0.01, 0.02, 2.3)), Eigen::Vector3s(0.66, -0.46, 0.3)};
}

MotionPrior driving_motion(const Timestamps& timestamps) {
  return {
      .start_time = timestamps.min,
      .linear_velocity = Eigen::Vector3s(1.5, 0.2, 0.0),
      .acceleration = Eigen::Vector3s(0.5, 0.0, 0.1),
      .angular_velocity = Eigen::Vector3s(0.05, -0.03, 1.0),
  };
}
} // namespace

TEST_CASE("deskew_scan: repeated stamps match the extrinsic then the per-point form", "[deskew]") {
  const bool per_stamp = GENERATE(false, true);
  const Timestamps timestamps = organised_timestamps(128, 450, Nsec(100'000'000));
  REQUIRE(max_relative_difference(random_points(timestamps.per_point.size()), timestamps, driving_motion(timestamps),
                                  mount(), per_stamp) < TOLERANCE);
}

TEST_CASE("deskew_scan: all-distinct stamps match the extrinsic then the per-point form", "[deskew]") {
  const Timestamps timestamps = organised_timestamps(1, 20'000, Nsec(100'000'000));
  REQUIRE(max_relative_difference(random_points(timestamps.per_point.size()), timestamps, driving_motion(timestamps),
                                  mount(), false) < TOLERANCE);
}

TEST_CASE("deskew_scan: an identity extrinsic matches the per-point form", "[deskew]") {
  const bool per_stamp = GENERATE(false, true);
  const Timestamps timestamps = organised_timestamps(128, 450, Nsec(100'000'000));
  REQUIRE(max_relative_difference(random_points(timestamps.per_point.size()), timestamps, driving_motion(timestamps),
                                  Sophus::SE3s{}, per_stamp) < TOLERANCE);
}

TEST_CASE("deskew_scan: zero angular velocity matches the extrinsic then the per-point form", "[deskew]") {
  const bool per_stamp = GENERATE(false, true);
  const Timestamps timestamps = organised_timestamps(32, 1024, Nsec(100'000'000));
  MotionPrior motion = driving_motion(timestamps);
  motion.angular_velocity.setZero();
  REQUIRE(max_relative_difference(random_points(timestamps.per_point.size()), timestamps, motion, mount(), per_stamp) <
          TOLERANCE);
}

TEST_CASE("deskew_scan: a rotation of 3 rad over the scan matches the extrinsic then the per-point form", "[deskew]") {
  const bool per_stamp = GENERATE(false, true);
  const Timestamps timestamps = organised_timestamps(64, 900, Nsec(300'000'000));
  MotionPrior motion = driving_motion(timestamps);
  motion.angular_velocity = Eigen::Vector3s(1.0, -2.0, 9.7);
  REQUIRE(max_relative_difference(random_points(timestamps.per_point.size()), timestamps, motion, mount(), per_stamp) <
          TOLERANCE);
}

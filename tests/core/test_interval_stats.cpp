#include "eigen_approx.hpp"

#include "rko_lio/core/util.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;
using rko_lio::core::IntervalStats;
using rko_lio::tests::EXACT_TOL;
using rko_lio::tests::TOL;

TEST_CASE("IntervalStats: default-constructed state is zero", "[interval_stats]") {
  IntervalStats stats;
  REQUIRE(stats.imu_count == 0);
  REQUIRE(stats.angular_velocity_sum.isZero());
  REQUIRE(stats.body_acceleration_sum.isZero());
  REQUIRE(stats.imu_acceleration_sum.isZero());
}

TEST_CASE("IntervalStats: single update", "[interval_stats]") {
  IntervalStats stats;
  const Eigen::Vector3s gyro(0.1, -0.2, 0.3);
  const Eigen::Vector3s accel(1.0, 2.0, 2.0);
  const Eigen::Vector3s compensated(0.0, 0.0, 0.0);
  stats.update(gyro, accel, compensated);

  REQUIRE(stats.imu_count == 1);
  REQUIRE(stats.angular_velocity_sum == gyro);
  REQUIRE(stats.imu_acceleration_sum == accel);
  REQUIRE(stats.body_acceleration_sum == compensated);
}

TEST_CASE("IntervalStats: reset zeros every field", "[interval_stats]") {
  IntervalStats stats;
  stats.update(Eigen::Vector3s(1, 2, 3), Eigen::Vector3s(4, 5, 6), Eigen::Vector3s(7, 8, 9));
  stats.update(Eigen::Vector3s(1, 1, 1), Eigen::Vector3s(2, 2, 2), Eigen::Vector3s(3, 3, 3));
  REQUIRE(stats.imu_count == 2);

  stats.reset();
  REQUIRE(stats.imu_count == 0);
  REQUIRE(stats.angular_velocity_sum.isZero());
  REQUIRE(stats.body_acceleration_sum.isZero());
  REQUIRE(stats.imu_acceleration_sum.isZero());
}

TEST_CASE("IntervalStats: sums accumulate across updates", "[interval_stats]") {
  IntervalStats stats;
  const Eigen::Vector3s g1(0.1, 0.2, 0.3);
  const Eigen::Vector3s g2(0.4, 0.5, 0.6);
  const Eigen::Vector3s a1(1.0, 0.0, 0.0);
  const Eigen::Vector3s a2(0.0, 1.0, 0.0);
  const Eigen::Vector3s c1(0.5, 0.0, 0.0);
  const Eigen::Vector3s c2(0.0, 0.5, 0.0);

  stats.update(g1, a1, c1);
  stats.update(g2, a2, c2);

  REQUIRE(stats.imu_count == 2);
  REQUIRE((stats.angular_velocity_sum - (g1 + g2)).norm() < EXACT_TOL);
  REQUIRE((stats.imu_acceleration_sum - (a1 + a2)).norm() < EXACT_TOL);
  REQUIRE((stats.body_acceleration_sum - (c1 + c2)).norm() < EXACT_TOL);
}

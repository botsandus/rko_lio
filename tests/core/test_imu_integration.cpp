#include "eigen_approx.hpp"
#include "rko_lio/core/lio.hpp"
#include "rko_lio/core/util.hpp"
#include "synthetic_clouds.hpp"
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

using Catch::Matchers::WithinAbs;
using rko_lio::core::GRAVITY_MAG;
using rko_lio::core::ImuControl;
using rko_lio::core::LIO;
using rko_lio::core::Nsec;
using rko_lio::core::Timestamps;
using rko_lio::core::TimestampVector;
using rko_lio::core::to_seconds;
using rko_lio::core::Vector3sVector;
using rko_lio::tests::approx_equal;
using rko_lio::tests::EXACT_TOL;
using rko_lio::tests::LOOSE_TOL;
using rko_lio::tests::make_hollow_cube;
using rko_lio::tests::register_cloud;
using rko_lio::tests::TOL;

namespace {
LIO::Config default_config() {
  LIO::Config cfg;
  cfg.min_range = 0.5;
  cfg.max_range = 50.0;
  cfg.voxel_size = 0.5;
  cfg.deskew = true;
  cfg.double_downsample = true;
  return cfg;
}

inline Nsec ns_from_seconds(double s) { return Nsec(static_cast<int64_t>(std::llround(s * 1e9))); }
} // namespace

TEST_CASE("IMU before first LiDAR is silently dropped", "[imu_integration]") {
  LIO lio(default_config());
  for (int i = 0; i < 5; ++i) {
    ImuControl m;
    m.time = ns_from_seconds(0.01 * (i + 1));
    m.acceleration = {0.0, 0.0, GRAVITY_MAG};
    m.angular_velocity = Eigen::Vector3s::Zero();
    lio.add_imu_measurement(m);
  }
  REQUIRE(lio.interval_stats.imu_count == 0);
}

TEST_CASE("IMU from the past is dropped", "[imu_integration]") {
  LIO lio(default_config());
  lio.lidar_state.time = ns_from_seconds(1.0);
  lio.imu.time = lio.lidar_state.time;

  ImuControl m;
  m.time = ns_from_seconds(1.1);
  m.acceleration = {0.0, 0.0, GRAVITY_MAG};
  m.angular_velocity = Eigen::Vector3s::Zero();
  lio.add_imu_measurement(m);

  const auto rotation = lio.imu.rotation;
  const auto time = lio.imu.time;
  const auto count = lio.interval_stats.imu_count;

  m.time = ns_from_seconds(0.5);
  lio.add_imu_measurement(m);

  REQUIRE(lio.interval_stats.imu_count == count);
  REQUIRE(approx_equal(lio.imu.rotation, rotation, EXACT_TOL));
  REQUIRE(lio.imu.time == time);
}

TEST_CASE("Static IMU at rest: gravity is compensated", "[imu_integration]") {
  LIO lio(default_config());
  // Bootstrap as if a first lidar scan had been registered.
  lio.lidar_state.time = ns_from_seconds(1.0);
  lio.imu.time = lio.lidar_state.time;

  constexpr int N = 50;
  for (int i = 0; i < N; ++i) {
    ImuControl m;
    m.time = ns_from_seconds(1.0 + 0.01 * (i + 1));
    m.acceleration = {0.0, 0.0, GRAVITY_MAG};
    m.angular_velocity = Eigen::Vector3s::Zero();
    lio.add_imu_measurement(m);
  }

  REQUIRE(lio.interval_stats.imu_count == N);
  REQUIRE(lio.interval_stats.body_acceleration_sum.norm() < TOL);
  REQUIRE_THAT(lio.interval_stats.imu_acceleration_sum.z(), WithinAbs(N * GRAVITY_MAG, LOOSE_TOL));
}

TEST_CASE("Pure rotation: gyro integration matches exp(omega*T)", "[imu_integration]") {
  LIO lio(default_config());
  lio.lidar_state.time = ns_from_seconds(1.0);
  lio.imu.time = lio.lidar_state.time;

  const Eigen::Vector3s omega(0.0, 0.0, 0.5);
  constexpr int N = 100;
  constexpr double dt = 0.01;
  const double T = N * dt;

  for (int i = 0; i < N; ++i) {
    ImuControl m;
    m.time = ns_from_seconds(1.0 + dt * (i + 1));
    m.angular_velocity = omega;
    m.acceleration = Eigen::Vector3s::Zero();
    lio.add_imu_measurement(m);
  }

  const Sophus::SO3s expected = Sophus::SO3s::exp(omega * T);
  REQUIRE(approx_equal(lio.imu.rotation, expected, TOL));
  REQUIRE_THAT(to_seconds(lio.imu.time), WithinAbs(1.0 + T, 1e-12));
}

TEST_CASE("Rolling IMU at rest: gravity is compensated in the integrated orientation", "[imu_integration]") {
  LIO lio(default_config());
  lio.lidar_state.time = ns_from_seconds(1.0);
  lio.imu.time = lio.lidar_state.time;

  // roll about x, so gravity leaves the z axis of the body; each sample reads gravity at the orientation it is
  // compensated with, the one before its own step
  const Eigen::Vector3s omega(0.5, 0.0, 0.0);
  constexpr int N = 100;
  constexpr double dt = 0.01;
  for (int i = 0; i < N; ++i) {
    ImuControl m;
    m.time = ns_from_seconds(1.0 + (dt * (i + 1)));
    m.angular_velocity = omega;
    m.acceleration = Sophus::SO3s::exp(omega * (dt * i)).inverse() * Eigen::Vector3s(0.0, 0.0, GRAVITY_MAG);
    lio.add_imu_measurement(m);
  }

  REQUIRE(lio.imu.rotation.log().norm() > 0.4);
  REQUIRE(lio.interval_stats.body_acceleration_sum.norm() < 1e-3);
}

TEST_CASE("Gyro bias is subtracted before integration", "[imu_integration]") {
  LIO lio(default_config());
  lio.lidar_state.time = ns_from_seconds(1.0);
  lio.imu.time = lio.lidar_state.time;
  lio.imu_bias.gyroscope = {0.1, -0.2, 0.05};

  const Eigen::Vector3s measured(0.5, 0.3, 0.1);
  constexpr int N = 50;
  for (int i = 0; i < N; ++i) {
    ImuControl m;
    m.time = ns_from_seconds(1.0 + 0.01 * (i + 1));
    m.angular_velocity = measured;
    m.acceleration = Eigen::Vector3s::Zero();
    lio.add_imu_measurement(m);
  }

  const Eigen::Vector3s expected = measured - lio.imu_bias.gyroscope;
  const Eigen::Vector3s avg = lio.interval_stats.angular_velocity_sum / lio.interval_stats.imu_count;
  REQUIRE(approx_equal(avg, expected, EXACT_TOL));
}

TEST_CASE("Extrinsic overload: identity extrinsic delegates to base-frame", "[imu_integration]") {
  LIO lio(default_config());
  lio.lidar_state.time = ns_from_seconds(1.0);
  lio.imu.time = lio.lidar_state.time;

  const Sophus::SE3s identity_extrinsic;
  const Eigen::Vector3s omega(0.1, 0.2, 0.3);

  constexpr int N = 10;
  for (int i = 0; i < N; ++i) {
    ImuControl m;
    m.time = ns_from_seconds(1.0 + 0.01 * (i + 1));
    m.angular_velocity = omega;
    m.acceleration = {0.0, 0.0, GRAVITY_MAG};
    lio.add_imu_measurement(identity_extrinsic, m);
  }

  // Identity-extrinsic short-circuits straight into the base-frame overload,
  // bypassing the extrinsic-overload's first-sample skip.
  REQUIRE(lio.interval_stats.imu_count == N);
  const Eigen::Vector3s avg_gyro = lio.interval_stats.angular_velocity_sum / lio.interval_stats.imu_count;
  REQUIRE(approx_equal(avg_gyro, omega, EXACT_TOL));
}

TEST_CASE("Extrinsic overload: pure-rotation extrinsic transforms omega", "[imu_integration]") {
  LIO lio(default_config());
  lio.lidar_state.time = ns_from_seconds(1.0);
  lio.imu.time = lio.lidar_state.time;

  const Sophus::SO3s R = Sophus::SO3s::rotX(M_PI / 2.0);
  const Sophus::SE3s extrinsic_imu2base(R, Eigen::Vector3s::Zero());

  const Eigen::Vector3s omega_imu(0.0, 0.0, 1.0);
  const Eigen::Vector3s expected_base = R * omega_imu;

  // The first extrinsic call is not integrated, so N+1 calls integrate N.
  constexpr int N = 20;
  for (int i = 0; i < N + 1; ++i) {
    ImuControl m;
    m.time = ns_from_seconds(1.0 + 0.01 * (i + 1));
    m.angular_velocity = omega_imu;
    m.acceleration = Eigen::Vector3s::Zero();
    lio.add_imu_measurement(extrinsic_imu2base, m);
  }

  REQUIRE(lio.interval_stats.imu_count == N);
  const Eigen::Vector3s avg_gyro = lio.interval_stats.angular_velocity_sum / lio.interval_stats.imu_count;
  REQUIRE(approx_equal(avg_gyro, expected_base, TOL));
}

TEST_CASE("Extrinsic overload: a rotated, offset IMU's lever arm is compensated", "[imu_integration]") {
  LIO lio(default_config());
  lio.lidar_state.time = ns_from_seconds(1.0);
  lio.imu.time = lio.lidar_state.time;

  // An IMU at r, rotated by R, on a base that turns at omega_k and accelerates at a_base. It measures R^T omega_k and
  // R^T (a_base + alpha_k x r + omega_k x (omega_k x r)), where alpha_k = (omega_k - omega_(k-1)) / dt. alpha differs
  // between the second and third sample, so the third sample compensates to a_base only if the LIO took the second
  // sample as its previous one.
  const Sophus::SO3s R = Sophus::SO3s::rotX(0.3) * Sophus::SO3s::rotZ(1.2);
  const Eigen::Vector3s r(0.4, -0.7, 1.5);
  const Sophus::SE3s extrinsic_imu2base(R, r);
  const Eigen::Vector3s a_base(0.3, -0.2, 0.1);
  constexpr double dt = 0.01;
  const std::array<Eigen::Vector3s, 3> omega{
      Eigen::Vector3s(0.1, -0.05, 0.5),
      Eigen::Vector3s(0.1, -0.05, 0.52),
      Eigen::Vector3s(0.12, -0.05, 0.5),
  };

  for (std::size_t k = 0; k < omega.size(); ++k) {
    const Eigen::Vector3s alpha =
        k == 0 ? Eigen::Vector3s::Zero() : Eigen::Vector3s((omega.at(k) - omega.at(k - 1)) / dt);
    ImuControl m;
    m.time = ns_from_seconds(1.0 + (dt * static_cast<double>(k + 1)));
    m.angular_velocity = R.inverse() * omega.at(k);
    m.acceleration = R.inverse() * (a_base + alpha.cross(r) + omega.at(k).cross(omega.at(k).cross(r)));
    lio.add_imu_measurement(extrinsic_imu2base, m);
  }

  // The first sample is not integrated, so two samples are. The second sample's alpha uses the first sample's angular
  // velocity, not zero.
  REQUIRE(lio.interval_stats.imu_count == 2);
  REQUIRE(approx_equal(lio.interval_stats.angular_velocity_sum, Eigen::Vector3s(omega[1] + omega[2]), TOL));
  REQUIRE(approx_equal(lio.interval_stats.imu_acceleration_sum, Eigen::Vector3s(2 * a_base), TOL));
}

TEST_CASE("Extrinsic overload: interleaved IMUs each keep their own history", "[imu_integration]") {
  LIO lio(default_config());
  lio.lidar_state.time = ns_from_seconds(1.0);
  lio.imu.time = lio.lidar_state.time;

  // Two IMUs with identity rotation sit at r_i on one base accelerating at a_base. IMU i has omega_i(t) = omega0_i +
  // alpha_i t, so it measures a_base + alpha_i x r_i + omega_i x (omega_i x r_i). The compensated acceleration equals
  // a_base only if alpha_i comes from IMU i's own previous sample. The sample periods differ, so the samples
  // interleave.
  struct Imu {
    Eigen::Vector3s r;
    Eigen::Vector3s omega0;
    Eigen::Vector3s alpha;
    double period;
  };
  const std::array<Imu, 2> imus{
      Imu{.r = {1.0, 0.0, 0.0}, .omega0 = {0.0, 0.0, 1.0}, .alpha = {0.0, 0.0, 2.0}, .period = 0.010},
      Imu{.r = {0.0, 1.0, 0.0}, .omega0 = {0.1, 0.0, -0.8}, .alpha = {0.5, 0.0, 3.0}, .period = 0.007},
  };
  const Eigen::Vector3s a_base(0.3, -0.2, 0.1);

  struct Sample {
    double time;
    std::size_t imu;
  };
  std::vector<Sample> samples;
  for (std::size_t imu = 0; imu < 2; ++imu) {
    for (int k = 1; k <= 5; ++k) {
      samples.push_back({.time = 1.0 + (imus.at(imu).period * k), .imu = imu});
    }
  }
  std::ranges::sort(samples, {}, &Sample::time);
  for (const Sample& sample : samples) {
    const Imu& imu = imus.at(sample.imu);
    const Eigen::Vector3s omega = imu.omega0 + (imu.alpha * (sample.time - 1.0));
    ImuControl m;
    m.time = ns_from_seconds(sample.time);
    m.angular_velocity = omega;
    m.acceleration = a_base + imu.alpha.cross(imu.r) + omega.cross(omega.cross(imu.r));
    lio.add_imu_measurement(Sophus::SE3s(Sophus::SO3s(), imu.r), m, sample.imu);
  }

  // Each IMU's first sample is not integrated, leaving 2 x 4 = 8 samples.
  REQUIRE(lio.interval_stats.imu_count == 8);
  REQUIRE(approx_equal(lio.interval_stats.imu_acceleration_sum, Eigen::Vector3s(8 * a_base), TOL));
}

TEST_CASE("The first payload anchors the IMU integration at its end", "[imu_integration]") {
  LIO lio((LIO::Config{}));
  const auto cloud = make_hollow_cube();
  constexpr double FIRST_SCAN_END = 10.0;
  register_cloud(lio, cloud,
                 Timestamps{
                     .min = ns_from_seconds(FIRST_SCAN_END),
                     .max = ns_from_seconds(FIRST_SCAN_END),
                     .per_point = TimestampVector(cloud.size(), ns_from_seconds(FIRST_SCAN_END)),
                 });

  // integrated from the payload end, the one sample turns the orientation by omega * dt
  const Eigen::Vector3s omega(0.0, 0.0, 0.5);
  constexpr double dt = 0.5;
  ImuControl m;
  m.time = ns_from_seconds(FIRST_SCAN_END + dt);
  m.acceleration = {0.0, 0.0, GRAVITY_MAG};
  m.angular_velocity = omega;
  lio.add_imu_measurement(m);
  REQUIRE(approx_equal(lio.imu.rotation, Sophus::SO3s::exp(omega * dt), TOL));
}

TEST_CASE("register_payload resets the IMU orientation to the optimized pose", "[imu_integration]") {
  // Mirrors the "Identity registration" test in test_register_payload.cpp: register the
  // same hollow-cube cloud twice with a static IMU between them, then assert that
  // the IMU orientation was re-anchored to lidar_state at scan end.
  LIO lio((LIO::Config{}));
  const auto cloud = make_hollow_cube();

  constexpr double FIRST_SCAN_END = 0.1;
  constexpr double DT = 1.0;
  constexpr double SECOND_SCAN_END = FIRST_SCAN_END + DT;

  // Build per-point timestamps for the first scan (linspace from 0 to FIRST_SCAN_END).
  TimestampVector ts1_per_point;
  ts1_per_point.reserve(cloud.size());
  for (size_t i = 0; i < cloud.size(); ++i) {
    const double frac = cloud.size() == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(cloud.size() - 1);
    ts1_per_point.emplace_back(ns_from_seconds(FIRST_SCAN_END * frac));
  }
  const Timestamps ts1{.min = ns_from_seconds(0.0), .max = ns_from_seconds(FIRST_SCAN_END), .per_point = ts1_per_point};
  register_cloud(lio, cloud, ts1);

  // Static IMU samples between the two scans. Use enough samples that interval_stats
  // is well-formed for the inter-scan filter.
  constexpr int N = 10;
  for (int i = 0; i < N; ++i) {
    ImuControl m;
    const double frac = static_cast<double>(i) / static_cast<double>(N - 1);
    m.time = ns_from_seconds(FIRST_SCAN_END + 0.05 + (DT - 0.1) * frac);
    m.acceleration = {0.0, 0.0, GRAVITY_MAG};
    m.angular_velocity = Eigen::Vector3s(0.0, 0.0, 0.05);
    lio.add_imu_measurement(m);
  }
  // the gyro turned the IMU orientation away from the lidar one, so the re-anchor below is observable
  REQUIRE(lio.imu.rotation.log().norm() > 0.02);

  // Single-instant timestamps for the second scan -> deskewing is identity.
  const Timestamps ts2{.min = ns_from_seconds(SECOND_SCAN_END),
                       .max = ns_from_seconds(SECOND_SCAN_END),
                       .per_point = TimestampVector(cloud.size(), ns_from_seconds(SECOND_SCAN_END))};
  register_cloud(lio, cloud, ts2);

  REQUIRE_THAT(to_seconds(lio.lidar_state.time), WithinAbs(SECOND_SCAN_END, 1e-9));
  REQUIRE(approx_equal(lio.imu.rotation, lio.lidar_state.pose.so3(), EXACT_TOL));
  REQUIRE_THAT(to_seconds(lio.imu.time), WithinAbs(to_seconds(lio.lidar_state.time), 1e-12));
}

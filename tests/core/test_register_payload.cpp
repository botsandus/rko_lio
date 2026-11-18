#include "eigen_approx.hpp"
#include "rko_lio/core/error.hpp"
#include "rko_lio/core/lio.hpp"
#include "synthetic_clouds.hpp"
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>

using Catch::Matchers::WithinAbs;
using rko_lio::core::GRAVITY_MAG;
using rko_lio::core::ImuControl;
using rko_lio::core::InputError;
using rko_lio::core::LIO;
using rko_lio::core::Nsec;
using rko_lio::core::RegistrationError;
using rko_lio::core::Scalar;
using rko_lio::core::Timestamps;
using rko_lio::core::TimestampVector;
using rko_lio::core::to_seconds;
using rko_lio::core::Vector3sVector;
using rko_lio::tests::approx_equal;
using rko_lio::tests::EXACT_TOL;
using rko_lio::tests::LOOSE_TOL;
using rko_lio::tests::make_hollow_cube;
using rko_lio::tests::register_cloud;

namespace {
inline Nsec ns_from_seconds(double s) { return Nsec(static_cast<int64_t>(std::llround(s * 1e9))); }
} // namespace

namespace {
// dt between consecutive scans, used to derive the IMU values that produce a
// given initial-guess pose. With v0 = 0 the kinematic model collapses to
//   tau.head = a * DT^2 / 2,  tau.tail = omega * DT,  initial_guess = SE3::exp(tau).
constexpr double DT = 1.0;
constexpr double FIRST_SCAN_END = 0.1;
constexpr double SECOND_SCAN_END = FIRST_SCAN_END + DT;

Timestamps linspace_timestamps(size_t n, double t_start, double t_end) {
  TimestampVector ts;
  ts.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const double frac = n == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(n - 1);
    ts.emplace_back(ns_from_seconds(t_start + (t_end - t_start) * frac));
  }
  const Nsec t_start_ns = ns_from_seconds(t_start);
  const Nsec t_end_ns = ns_from_seconds(t_end);
  return Timestamps{
      .min = std::min(t_start_ns, t_end_ns), .max = std::max(t_start_ns, t_end_ns), .per_point = std::move(ts)};
}

// Single-instant timestamps for the second scan: deskewing reduces to identity
// when every point shares end_time.
Timestamps instant_timestamps(size_t n, double t) {
  return Timestamps{
      .min = ns_from_seconds(t), .max = ns_from_seconds(t), .per_point = TimestampVector(n, ns_from_seconds(t))};
}

void feed_imu(LIO& lio,
              double t_start,
              double t_end,
              int n,
              const Eigen::Vector3s& acceleration,
              const Eigen::Vector3s& angular_velocity) {
  for (int i = 0; i < n; ++i) {
    ImuControl m;
    const double frac = n == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(n - 1);
    m.time = ns_from_seconds(t_start + (t_end - t_start) * frac);
    m.acceleration = acceleration;
    m.angular_velocity = angular_velocity;
    lio.add_imu_measurement(m);
  }
}

void feed_static_imu(LIO& lio, double t_start, double t_end, int n) {
  feed_imu(lio, t_start, t_end, n, {0.0, 0.0, GRAVITY_MAG}, Eigen::Vector3s::Zero());
}

// Same as feed_imu but with per-sample Gaussian noise added to accel and gyro.
// Seed is fixed at the call site so the test is deterministic.
void feed_imu_noisy(LIO& lio,
                    double t_start,
                    double t_end,
                    int n,
                    const Eigen::Vector3s& acceleration,
                    const Eigen::Vector3s& angular_velocity,
                    double accel_stddev,
                    double gyro_stddev,
                    uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> a_noise(0.0, accel_stddev);
  std::normal_distribution<double> g_noise(0.0, gyro_stddev);
  for (int i = 0; i < n; ++i) {
    ImuControl m;
    const double frac = n == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(n - 1);
    m.time = ns_from_seconds(t_start + (t_end - t_start) * frac);
    m.acceleration = acceleration + Eigen::Vector3s(a_noise(rng), a_noise(rng), a_noise(rng));
    m.angular_velocity = angular_velocity + Eigen::Vector3s(g_noise(rng), g_noise(rng), g_noise(rng));
    lio.add_imu_measurement(m);
  }
}

// Noise levels for the [!mayfail] robustness variants. Small enough that ICP
// should still close the gap to ground truth; large enough to perturb the
// IMU-derived initial guess noticeably.
constexpr double NOISY_ACCEL_STDDEV = 0.05; // m/s^2
constexpr double NOISY_GYRO_STDDEV = 5e-3;  // rad/s
} // namespace

TEST_CASE("First scan: empty map -> populated, single pose at lidar_state.time", "[register_payload]") {
  LIO lio((LIO::Config{}));
  const auto cloud = make_hollow_cube();
  const auto ts = linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END);

  REQUIRE(lio.map.empty());
  REQUIRE(lio.lidar_state.time == Nsec{0});

  register_cloud(lio, cloud, ts);

  REQUIRE_FALSE(lio.map.empty());
  REQUIRE_THAT(to_seconds(lio.lidar_state.time), WithinAbs(FIRST_SCAN_END, 1e-9));
  REQUIRE(approx_equal(lio.lidar_state.pose, Sophus::SE3s{}, EXACT_TOL));
}

TEST_CASE("Identity registration: same cloud twice -> pose ~= identity", "[register_payload]") {
  LIO lio((LIO::Config{}));
  const auto cloud = make_hollow_cube();

  register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END));
  feed_static_imu(lio, FIRST_SCAN_END + 0.05, SECOND_SCAN_END - 0.05, 10);
  register_cloud(lio, cloud, instant_timestamps(cloud.size(), SECOND_SCAN_END));

  REQUIRE_THAT(to_seconds(lio.lidar_state.time), WithinAbs(SECOND_SCAN_END, 1e-9));
  REQUIRE(lio.lidar_state.pose.translation().norm() < 1e-3);
  REQUIRE(lio.lidar_state.pose.so3().log().norm() < 1e-3);
}

TEST_CASE("payload_stats: the scan's points, the ICP's input and last iteration, and the IMU samples it had",
          "[register_payload]") {
  LIO lio((LIO::Config{}));
  const auto cloud = make_hollow_cube();

  register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END));
  REQUIRE(lio.payload_stats.input_points == cloud.size()); // the first scan seeds the map: no ICP
  REQUIRE(lio.payload_stats.icp_iterations == 0);

  feed_static_imu(lio, FIRST_SCAN_END + 0.05, SECOND_SCAN_END - 0.05, 10);
  register_cloud(lio, cloud, instant_timestamps(cloud.size(), SECOND_SCAN_END));
  REQUIRE(lio.payload_stats.input_points == cloud.size());
  REQUIRE(lio.payload_stats.imu_samples == 10);
  REQUIRE(lio.payload_stats.keypoints > 0);
  REQUIRE(lio.payload_stats.keypoints <= cloud.size());
  REQUIRE(lio.payload_stats.icp_iterations >= 1);
  REQUIRE(lio.payload_stats.icp_iterations <= lio.config.max_iterations);
  REQUIRE(lio.payload_stats.correspondences > 0);
  REQUIRE(std::cmp_less_equal(lio.payload_stats.correspondences, lio.payload_stats.keypoints));
  REQUIRE(lio.payload_stats.icp_cost < 1e-4); // the same cloud again: it aligns

  register_cloud(lio, cloud, instant_timestamps(cloud.size(), FIRST_SCAN_END)); // not after the previous scan: skipped
  REQUIRE(lio.payload_stats.input_points == cloud.size());
  REQUIRE(lio.payload_stats.icp_iterations == 0);
}

TEST_CASE("deskew = false: register_payload still produces a valid pose", "[register_payload]") {
  LIO::Config cfg{};
  cfg.deskew = false;
  LIO lio(cfg);
  const auto cloud = make_hollow_cube();

  register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END));
  feed_static_imu(lio, FIRST_SCAN_END + 0.05, SECOND_SCAN_END - 0.05, 10);
  register_cloud(lio, cloud, instant_timestamps(cloud.size(), SECOND_SCAN_END));

  REQUIRE_THAT(to_seconds(lio.lidar_state.time), WithinAbs(SECOND_SCAN_END, 1e-9));
  REQUIRE(lio.lidar_state.pose.translation().norm() < 1e-3);
  REQUIRE(lio.lidar_state.pose.so3().log().norm() < 1e-3);
}

TEST_CASE("Pure translation: recover +1m in x", "[register_payload]") {
  LIO lio((LIO::Config{}));
  const auto cloud = make_hollow_cube();

  register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END));

  // Body accel a such that a * DT^2 / 2 = 1m in x  =>  a = 2 m/s^2. Add gravity for
  // the raw IMU value so compensated body accel comes out to (a, 0, 0).
  const Eigen::Vector3s t(1.0, 0.0, 0.0);
  const Eigen::Vector3s a_body = 2.0 * t / (DT * DT);
  const Eigen::Vector3s imu_accel = a_body + Eigen::Vector3s{0.0, 0.0, GRAVITY_MAG};
  feed_imu(lio, FIRST_SCAN_END + 0.05, SECOND_SCAN_END - 0.05, 10, imu_accel, Eigen::Vector3s::Zero());

  Vector3sVector cloud2;
  cloud2.reserve(cloud.size());
  for (const auto& p : cloud) {
    cloud2.emplace_back(p - t);
  }
  register_cloud(lio, cloud2, instant_timestamps(cloud2.size(), SECOND_SCAN_END));

  const auto& pose = lio.lidar_state.pose;
  REQUIRE(approx_equal(pose.translation(), t, 1e-3));
  REQUIRE(pose.so3().log().norm() < 1e-3);
}

TEST_CASE("Pure rotation: recover 5 deg yaw", "[register_payload]") {
  LIO lio((LIO::Config{}));
  const auto cloud = make_hollow_cube();

  register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END));

  // omega * DT = yaw (5 deg) about z  =>  omega = yaw/DT rad/s.
  const double yaw = 5.0 * M_PI / 180.0;
  const Eigen::Vector3s omega(0.0, 0.0, yaw / DT);
  feed_imu(lio, FIRST_SCAN_END + 0.05, SECOND_SCAN_END - 0.05, 10, {0.0, 0.0, GRAVITY_MAG}, omega);

  const Sophus::SO3s R = Sophus::SO3s::rotZ(yaw);
  const Sophus::SO3s R_inv = R.inverse();
  Vector3sVector cloud2;
  cloud2.reserve(cloud.size());
  for (const auto& p : cloud) {
    cloud2.emplace_back(R_inv * p);
  }
  register_cloud(lio, cloud2, instant_timestamps(cloud2.size(), SECOND_SCAN_END));

  const auto& pose = lio.lidar_state.pose;
  REQUIRE(approx_equal(pose.so3(), R, 1e-3));
  REQUIRE(pose.translation().norm() < 1e-3);
}

TEST_CASE("Gap in the lidar stream: prior extrapolates instead of throwing", "[register_payload]") {
  LIO lio((LIO::Config{}));
  const auto cloud = make_hollow_cube();

  register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END));

  // a gap longer than the one second tolerance, with the imu still streaming. yawing about the gravity axis keeps the
  // gravity compensation exact, so the body acceleration in x turns into a real translation as well as a rotation
  constexpr double SECOND_SCAN_START = FIRST_SCAN_END + 1.5;
  feed_imu(lio, FIRST_SCAN_END, SECOND_SCAN_START, 150, {2.0, 0.0, GRAVITY_MAG}, {0.0, 0.0, 0.4});

  // the imu integration is where that motion actually ended up, so put the cloud there and let the prior find it
  const Sophus::SE3s ground_truth = lio.imu_state.pose;
  REQUIRE(ground_truth.translation().norm() > 1.0);
  REQUIRE(ground_truth.so3().log().norm() > 0.1);

  const Sophus::SE3s into_ground_truth_frame = ground_truth.inverse();
  Vector3sVector cloud2;
  cloud2.reserve(cloud.size());
  for (const auto& p : cloud) {
    cloud2.emplace_back(into_ground_truth_frame * p);
  }
  REQUIRE_NOTHROW(register_cloud(lio, cloud2, instant_timestamps(cloud2.size(), SECOND_SCAN_START)));

  REQUIRE_THAT(to_seconds(lio.lidar_state.time), WithinAbs(SECOND_SCAN_START, 1e-9));
  REQUIRE(approx_equal(lio.lidar_state.pose, ground_truth, 1e-2));
}

TEST_CASE("Gap in the lidar stream: second scan has real per-point spread, not instant", "[register_payload]") {
  LIO lio((LIO::Config{}));
  const auto cloud = make_hollow_cube(2.0, 5.0);

  register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END));

  constexpr double SECOND_SCAN_START = FIRST_SCAN_END + 1.5;
  constexpr double SECOND_SCAN_SPAN_END = SECOND_SCAN_START + 0.1;
  const Eigen::Vector3s a_body(2.0, 0.0, 0.0);
  const Eigen::Vector3s omega(0.0, 0.0, 0.4);
  const Eigen::Vector3s imu_accel = a_body + Eigen::Vector3s{0.0, 0.0, GRAVITY_MAG};
  feed_imu(lio, FIRST_SCAN_END, SECOND_SCAN_START, 150, imu_accel, omega);

  // one imu sample per point, at that point's own timestamp; imu_state right after is a true
  // per-sample-integrated pose for that point, independent of deskew's own (coarser) model
  const Timestamps ts2 = linspace_timestamps(cloud.size(), SECOND_SCAN_START, SECOND_SCAN_SPAN_END);
  Vector3sVector cloud2(cloud.size());
  for (size_t i = 0; i < cloud.size(); ++i) {
    ImuControl m;
    m.time = ts2.per_point[i];
    m.acceleration = imu_accel;
    m.angular_velocity = omega;
    lio.add_imu_measurement(m);
    cloud2[i] = lio.imu_state.pose.inverse() * cloud[i];
  }
  const Sophus::SE3s ground_truth = lio.imu_state.pose;
  REQUIRE(ground_truth.translation().norm() > 1.0);
  REQUIRE(ground_truth.so3().log().norm() > 0.1);

  REQUIRE_NOTHROW(register_cloud(lio, cloud2, ts2));

  REQUIRE_THAT(to_seconds(lio.lidar_state.time), WithinAbs(SECOND_SCAN_SPAN_END, 1e-9));
  REQUIRE(approx_equal(lio.lidar_state.pose, ground_truth, 1e-2));
}

TEST_CASE("Full SE(3): translation + rotation combined", "[register_payload]") {
  LIO lio((LIO::Config{}));
  const auto cloud = make_hollow_cube();

  register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END));

  // Combined yaw + acceleration
  const double yaw = 3.0 * M_PI / 180.0;
  const Eigen::Vector3s t(0.5, 0.3, 0.0);
  const Eigen::Vector3s a_body = 2.0 * t / (DT * DT);
  const Eigen::Vector3s imu_accel = a_body + Eigen::Vector3s{0.0, 0.0, GRAVITY_MAG};
  const Eigen::Vector3s omega(0.0, 0.0, yaw / DT);
  feed_imu(lio, FIRST_SCAN_END + 0.05, SECOND_SCAN_END - 0.05, 10, imu_accel, omega);

  const Sophus::SE3s T_expected(Sophus::SO3s::rotZ(yaw), t);
  const Sophus::SE3s T_inv = T_expected.inverse();
  Vector3sVector cloud2;
  cloud2.reserve(cloud.size());
  for (const auto& p : cloud) {
    cloud2.emplace_back(T_inv * p);
  }
  register_cloud(lio, cloud2, instant_timestamps(cloud2.size(), SECOND_SCAN_END));

  REQUIRE(approx_equal(lio.lidar_state.pose, T_expected, 1e-2));
}

// ===========================================================================
// Noisy variants. The IMU-derived initial guess is perturbed by Gaussian noise;
// ICP has to refine it. Tagged [!mayfail] so a borderline failure surfaces in
// the report without breaking the suite -- drop the tag to make these strict.
// ===========================================================================

TEST_CASE("Noisy: pure translation +1m in x", "[register_payload][!mayfail]") {
  LIO lio((LIO::Config{}));
  const auto cloud = make_hollow_cube();

  register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END));

  const Eigen::Vector3s t(1.0, 0.0, 0.0);
  const Eigen::Vector3s a_body = 2.0 * t / (DT * DT);
  const Eigen::Vector3s imu_accel = a_body + Eigen::Vector3s{0.0, 0.0, GRAVITY_MAG};
  feed_imu_noisy(lio, FIRST_SCAN_END + 0.05, SECOND_SCAN_END - 0.05, 10, imu_accel, Eigen::Vector3s::Zero(),
                 NOISY_ACCEL_STDDEV, NOISY_GYRO_STDDEV, /*seed=*/1u);

  Vector3sVector cloud2;
  cloud2.reserve(cloud.size());
  for (const auto& p : cloud) {
    cloud2.emplace_back(p - t);
  }
  register_cloud(lio, cloud2, instant_timestamps(cloud2.size(), SECOND_SCAN_END));

  const auto& pose = lio.lidar_state.pose;
  CAPTURE(pose.translation().transpose(), pose.so3().log().transpose());
  CHECK(approx_equal(pose.translation(), t, 1e-3));
  CHECK(pose.so3().log().norm() < 1e-3);
}

TEST_CASE("Noisy: pure rotation 5 deg yaw", "[register_payload][!mayfail]") {
  LIO lio((LIO::Config{}));
  const auto cloud = make_hollow_cube();

  register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END));

  const double yaw = 5.0 * M_PI / 180.0;
  const Eigen::Vector3s omega(0.0, 0.0, yaw / DT);
  feed_imu_noisy(lio, FIRST_SCAN_END + 0.05, SECOND_SCAN_END - 0.05, 10, {0.0, 0.0, GRAVITY_MAG}, omega,
                 NOISY_ACCEL_STDDEV, NOISY_GYRO_STDDEV,
                 /*seed=*/2u);

  const Sophus::SO3s R = Sophus::SO3s::rotZ(yaw);
  const Sophus::SO3s R_inv = R.inverse();
  Vector3sVector cloud2;
  cloud2.reserve(cloud.size());
  for (const auto& p : cloud) {
    cloud2.emplace_back(R_inv * p);
  }
  register_cloud(lio, cloud2, instant_timestamps(cloud2.size(), SECOND_SCAN_END));

  const auto& pose = lio.lidar_state.pose;
  CAPTURE(pose.translation().transpose(), pose.so3().log().transpose());
  CHECK(approx_equal(pose.so3(), R, 1e-3));
  CHECK(pose.translation().norm() < 1e-3);
}

TEST_CASE("Noisy: full SE(3) translation + rotation", "[register_payload][!mayfail]") {
  LIO lio((LIO::Config{}));
  const auto cloud = make_hollow_cube();

  register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END));

  const double yaw = 3.0 * M_PI / 180.0;
  const Eigen::Vector3s t(0.5, 0.3, 0.0);
  const Eigen::Vector3s a_body = 2.0 * t / (DT * DT);
  const Eigen::Vector3s imu_accel = a_body + Eigen::Vector3s{0.0, 0.0, GRAVITY_MAG};
  const Eigen::Vector3s omega(0.0, 0.0, yaw / DT);
  feed_imu_noisy(lio, FIRST_SCAN_END + 0.05, SECOND_SCAN_END - 0.05, 10, imu_accel, omega, NOISY_ACCEL_STDDEV,
                 NOISY_GYRO_STDDEV, /*seed=*/3u);

  const Sophus::SE3s T_expected(Sophus::SO3s::rotZ(yaw), t);
  const Sophus::SE3s T_inv = T_expected.inverse();
  Vector3sVector cloud2;
  cloud2.reserve(cloud.size());
  for (const auto& p : cloud) {
    cloud2.emplace_back(T_inv * p);
  }
  register_cloud(lio, cloud2, instant_timestamps(cloud2.size(), SECOND_SCAN_END));

  CAPTURE(lio.lidar_state.pose.translation().transpose(), lio.lidar_state.pose.so3().log().transpose());
  CHECK(approx_equal(lio.lidar_state.pose, T_expected, 1e-2));
}

TEST_CASE("register_payload: timestamp count mismatch throws instead of UB", "[register_payload]") {
  LIO lio((LIO::Config{}));
  const auto cloud = make_hollow_cube();
  REQUIRE_THROWS_AS(register_cloud(lio, cloud, Timestamps{}), InputError);
  auto ts = linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END);
  ts.per_point.pop_back();
  REQUIRE_THROWS_AS(register_cloud(lio, cloud, ts), InputError);
}

TEST_CASE("register_payload: a scan without one extrinsic per cloud throws, its own stats set", "[register_payload]") {
  LIO lio((LIO::Config{}));
  const auto cloud = make_hollow_cube();
  register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END));

  const Vector3sVector part(cloud.begin(), cloud.begin() + 100);
  const std::array<Timestamps, 1> timestamps{linspace_timestamps(part.size(), FIRST_SCAN_END, SECOND_SCAN_END)};
  REQUIRE_THROWS_AS(lio.register_payload(std::vector<Vector3sVector>{part}, timestamps, {}), InputError);
  REQUIRE(lio.payload_stats.input_points == part.size());
}

TEST_CASE("register_payload: each cloud's extrinsic brings it to the base frame", "[register_payload]") {
  LIO::Config config;
  config.deskew = GENERATE(false, true);
  LIO lio(config);
  const auto cube = make_hollow_cube();
  const std::array<Sophus::SE3s, 2> lidar2base{
      Sophus::SE3s(Sophus::SO3s::rotZ(0.5), Eigen::Vector3s(0.6, 0.4, 0.3)),
      Sophus::SE3s(Sophus::SO3s::rotX(-0.2), Eigen::Vector3s(0.6, -0.4, 0.3)),
  };
  std::vector<Vector3sVector> lidar_clouds;
  for (const Sophus::SE3s& mount : lidar2base) {
    Vector3sVector& lidar_cloud = lidar_clouds.emplace_back();
    for (const auto& p : cube) {
      lidar_cloud.emplace_back(mount.inverse() * p);
    }
  }

  // the first scan seeds the map without deskewing, the second one is registered
  for (const double scan_end : {FIRST_SCAN_END, SECOND_SCAN_END}) {
    const std::array<Timestamps, 2> timestamps{
        instant_timestamps(cube.size(), scan_end),
        instant_timestamps(cube.size(), scan_end),
    };
    const std::vector<Vector3sVector> base_clouds = lio.register_payload(lidar_clouds, timestamps, lidar2base);
    REQUIRE(base_clouds.size() == 2);
    for (const Vector3sVector& base_cloud : base_clouds) {
      REQUIRE(std::ranges::equal(base_cloud, cube, [](const Eigen::Vector3s& base_point, const Eigen::Vector3s& point) {
        return approx_equal(base_point, point, LOOSE_TOL);
      }));
    }
  }
}

TEST_CASE("register_payload: zero correspondences throws RegistrationError", "[register_payload]") {
  LIO lio((LIO::Config{}));
  const auto cloud = make_hollow_cube();
  register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END));
  const auto pose_before = lio.lidar_state.pose;
  const auto time_before = lio.lidar_state.time;

  feed_static_imu(lio, FIRST_SCAN_END, SECOND_SCAN_END, 10);
  auto cloud2 = cloud;
  for (auto& p : cloud2) {
    p.x() += 20.0;
  }

  REQUIRE_THROWS_AS(register_cloud(lio, cloud2, instant_timestamps(cloud2.size(), SECOND_SCAN_END)), RegistrationError);
  REQUIRE(approx_equal(lio.lidar_state.pose, pose_before, EXACT_TOL));
  REQUIRE(lio.lidar_state.time == time_before);
}

TEST_CASE("reset after RegistrationError: next scan bootstraps like a first scan", "[register_payload]") {
  LIO::Config cfg{};
  cfg.initialization_phase = true;
  LIO lio(cfg);
  const auto cloud = make_hollow_cube();

  lio.restart();
  REQUIRE(lio.reset_count == 1);
  REQUIRE_FALSE(lio.config.initialization_phase);

  register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END));
  REQUIRE_FALSE(lio.map.empty());

  feed_static_imu(lio, FIRST_SCAN_END, SECOND_SCAN_END, 10);
  auto cloud2 = cloud;
  for (auto& p : cloud2) {
    p.x() += 20.0;
  }
  REQUIRE_THROWS_AS(register_cloud(lio, cloud2, instant_timestamps(cloud2.size(), SECOND_SCAN_END)), RegistrationError);

  lio.restart();
  REQUIRE(lio.reset_count == 2);
  REQUIRE(lio.map.empty());
  REQUIRE(lio.lidar_state.time == Nsec{0});
  REQUIRE(lio.interval_stats.imu_count == 0);
  REQUIRE(approx_equal(lio.lidar_state.pose, Sophus::SE3s{}, EXACT_TOL));

  constexpr double third_scan_end = SECOND_SCAN_END + DT;
  register_cloud(lio, cloud, instant_timestamps(cloud.size(), third_scan_end));
  REQUIRE_FALSE(lio.map.empty());
  REQUIRE_THAT(to_seconds(lio.lidar_state.time), WithinAbs(third_scan_end, 1e-9));
  REQUIRE(approx_equal(lio.lidar_state.pose, Sophus::SE3s{}, EXACT_TOL));
}

TEST_CASE("A scan that does not end after the previous one, or a fragment of a scan, is dropped",
          "[register_payload]") {
  LIO lio((LIO::Config{}));
  const auto cloud = make_hollow_cube();
  register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END));

  REQUIRE(register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END)).empty());
  REQUIRE(register_cloud(lio, cloud, linspace_timestamps(cloud.size(), -0.05, FIRST_SCAN_END - 0.05)).empty());

  const Vector3sVector fragment(cloud.begin(), cloud.begin() + static_cast<std::ptrdiff_t>(cloud.size() / 20));
  REQUIRE(
      register_cloud(lio, fragment, linspace_timestamps(fragment.size(), FIRST_SCAN_END + 0.098, FIRST_SCAN_END + 0.1))
          .empty());
  REQUIRE_THAT(to_seconds(lio.lidar_state.time), WithinAbs(FIRST_SCAN_END, 1e-9));

  REQUIRE_FALSE(register_cloud(lio, cloud, linspace_timestamps(cloud.size(), FIRST_SCAN_END, SECOND_SCAN_END)).empty());
  REQUIRE_THAT(to_seconds(lio.lidar_state.time), WithinAbs(SECOND_SCAN_END, 1e-9));
}

TEST_CASE("A full scan ending less than 10 ms after the previous one is dropped", "[register_payload]") {
  LIO lio((LIO::Config{}));
  const auto cloud = make_hollow_cube();
  register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END));

  REQUIRE(register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.009, FIRST_SCAN_END + 0.009)).empty());
  REQUIRE_THAT(to_seconds(lio.lidar_state.time), WithinAbs(FIRST_SCAN_END, 1e-9));

  REQUIRE_FALSE(register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.011, FIRST_SCAN_END + 0.011)).empty());
  REQUIRE_THAT(to_seconds(lio.lidar_state.time), WithinAbs(FIRST_SCAN_END + 0.011, 1e-9));
}

TEST_CASE("restart keeps the IMU bias estimates", "[register_payload]") {
  LIO lio((LIO::Config{}));
  lio.imu_bias.accelerometer = {0.1, -0.2, 0.3};
  lio.imu_bias.gyroscope = {0.01, 0.02, -0.03};
  lio.restart();
  REQUIRE(lio.imu_bias.accelerometer == Eigen::Vector3s(0.1, -0.2, 0.3));
  REQUIRE(lio.imu_bias.gyroscope == Eigen::Vector3s(0.01, 0.02, -0.03));
}

TEST_CASE("flat: z, roll and pitch stay at their initial values", "[register_payload]") {
  LIO::Config config;
  config.flat = true;
  LIO lio(config);
  const auto cloud = make_hollow_cube();
  register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END));

  const double yaw = 3.0 * M_PI / 180.0;
  const Eigen::Vector3s t(0.5, 0.3, 0.2);
  const Eigen::Vector3s imu_accel = 2.0 * t / (DT * DT) + Eigen::Vector3s{0.0, 0.0, GRAVITY_MAG};
  feed_imu(lio, FIRST_SCAN_END + 0.05, SECOND_SCAN_END - 0.05, 10, imu_accel,
           Eigen::Vector3s(0.01, 0.02, static_cast<Scalar>(yaw / DT)));

  const Sophus::SE3s T_inv = Sophus::SE3s(Sophus::SO3s::rotZ(static_cast<Scalar>(yaw)), t).inverse();
  Vector3sVector cloud2;
  cloud2.reserve(cloud.size());
  for (const auto& p : cloud) {
    cloud2.emplace_back(T_inv * p);
  }
  register_cloud(lio, cloud2, instant_timestamps(cloud2.size(), SECOND_SCAN_END));

  REQUIRE(
      approx_equal(lio.lidar_state.pose, Sophus::SE3s(Sophus::SO3s::rotZ(yaw), Eigen::Vector3s(0.5, 0.3, 0.0)), 1e-2));
}

TEST_CASE("flat: the tilt from initialization_phase compensates gravity and stays out of the pose",
          "[register_payload]") {
  LIO::Config config;
  config.flat = true;
  config.initialization_phase = true;
  LIO lio(config);
  const auto cloud = make_hollow_cube();
  const Eigen::Vector3s tilted_gravity = Sophus::SO3s::rotX(0.1) * Eigen::Vector3s(0.0, 0.0, GRAVITY_MAG);

  register_cloud(lio, cloud, linspace_timestamps(cloud.size(), 0.0, FIRST_SCAN_END));
  feed_imu(lio, FIRST_SCAN_END + 0.05, SECOND_SCAN_END - 0.05, 10, tilted_gravity, Eigen::Vector3s::Zero());
  register_cloud(lio, cloud, instant_timestamps(cloud.size(), SECOND_SCAN_END));
  feed_imu(lio, SECOND_SCAN_END + 0.05, SECOND_SCAN_END + DT - 0.05, 10, tilted_gravity, Eigen::Vector3s::Zero());

  REQUIRE(approx_equal(lio.lidar_state.pose, Sophus::SE3s{}, EXACT_TOL));
  REQUIRE(lio.interval_stats.body_acceleration_sum.norm() < EXACT_TOL);
}

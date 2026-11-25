#include "rko_lio/ros/base_node.hpp"
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstddef>
#include <optional>
#include <vector>

using rko_lio::core::Nsec;
using rko_lio::ros::ImuInput;
using rko_lio::ros::imus_cover;
using namespace std::chrono_literals;

namespace {
// An IMU with samples_before samples stamped 500 ms, followed by one at newest_time when newest_time is set.
ImuInput imu(const std::optional<Nsec> newest_time, const std::size_t samples_before = 0) {
  ImuInput in{.newest_time = newest_time};
  in.buffer.resize(samples_before, {.time = 500ms});
  if (newest_time) {
    in.buffer.push_back({.time = *newest_time});
  }
  return in;
}
} // namespace

TEST_CASE("imus_cover: one IMU covers an end it has a sample past, strictly", "[imus_cover]") {
  REQUIRE(imus_cover(std::vector{imu(1s)}, 0ns, 900ms, 10));
  REQUIRE_FALSE(imus_cover(std::vector{imu(1s)}, 0ns, 1s, 10));
}

TEST_CASE("imus_cover: a payload waits for a lagging IMU that is not stale", "[imus_cover]") {
  REQUIRE_FALSE(imus_cover(std::vector{imu(1s), imu(950ms)}, 0ns, 970ms, 10));
  REQUIRE(imus_cover(std::vector{imu(1s), imu(950ms)}, 0ns, 900ms, 10));
}

TEST_CASE("imus_cover: with 10 samples since fed_until before the end, one IMU past the end is enough",
          "[imus_cover]") {
  // The first IMU has 5 samples at 500 ms, the lagging IMU 4 at 500 ms and one at 950 ms: 10 samples before the end at
  // 970 ms. The first IMU's newest sample, at 1 s, is past the end.
  REQUIRE(imus_cover(std::vector{imu(1s, 5), imu(950ms, 4)}, 0ns, 970ms, 10));
  REQUIRE_FALSE(imus_cover(std::vector{imu(1s, 5), imu(950ms, 3)}, 0ns, 970ms, 10));
  // With fed_until at 600 ms the samples at 500 ms do not count, because they were fed with an earlier payload.
  REQUIRE_FALSE(imus_cover(std::vector{imu(1s, 5), imu(950ms, 4)}, 600ms, 970ms, 10));
  // without a sample minimum, as offline, the lagging IMU is waited for
  REQUIRE_FALSE(imus_cover(std::vector{imu(1s, 5), imu(950ms, 4)}, 0ns, 970ms, std::nullopt));
  // no IMU past the end, however many samples
  REQUIRE_FALSE(imus_cover(std::vector{imu(950ms, 20), imu(940ms, 20)}, 0ns, 970ms, 10));
}

TEST_CASE("imus_cover: an IMU more than 0.25 s behind the newest is not waited for", "[imus_cover]") {
  REQUIRE(imus_cover(std::vector{imu(1s), imu(749ms)}, 0ns, 900ms, 10));
  REQUIRE_FALSE(imus_cover(std::vector{imu(1s), imu(750ms)}, 0ns, 900ms, 10));
}

TEST_CASE("imus_cover: an IMU without a sample is not waited for, and no IMU covers nothing", "[imus_cover]") {
  REQUIRE(imus_cover(std::vector{imu(1s), imu(std::nullopt)}, 0ns, 900ms, 10));
  REQUIRE_FALSE(imus_cover(std::vector{imu(std::nullopt), imu(std::nullopt)}, 0ns, 900ms, 10));
}

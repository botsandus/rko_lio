#include "rko_lio/core/error.hpp"
#include "rko_lio/ros/utils/point_cloud_read.hpp"
#include "rko_lio/ros/utils/time.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace std::chrono_literals;
using rko_lio::core::Nsec;
using rko_lio::core::TimestampVector;
using rko_lio::ros::utils::point_cloud2_to_scan;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

constexpr Nsec HEADER_STAMP = 1'700'000'000'123'456'789ns;
constexpr float NOT_FINITE = std::numeric_limits<float>::quiet_NaN();

// one point per x, its time field t of the given datatype
template <typename T>
PointCloud2 make_cloud(const std::vector<float>& xs, const std::vector<T>& times, const std::uint8_t datatype) {
  PointCloud2 msg;
  msg.header.stamp = rko_lio::ros::utils::to_ros_time(HEADER_STAMP);
  sensor_msgs::PointCloud2Modifier modifier(msg);
  modifier.setPointCloud2Fields(4, "x", 1, PointField::FLOAT32, "y", 1, PointField::FLOAT32, "z", 1,
                                PointField::FLOAT32, "t", 1, datatype);
  modifier.resize(xs.size());
  sensor_msgs::PointCloud2Iterator<float> msg_x(msg, "x");
  sensor_msgs::PointCloud2Iterator<float> msg_y(msg, "y");
  sensor_msgs::PointCloud2Iterator<float> msg_z(msg, "z");
  sensor_msgs::PointCloud2Iterator<T> msg_time(msg, "t");
  for (std::size_t i = 0; i < xs.size(); ++i, ++msg_x, ++msg_y, ++msg_z, ++msg_time) {
    *msg_x = xs[i];
    *msg_y = 1;
    *msg_z = 2;
    *msg_time = times[i];
  }
  return msg;
}
} // namespace

TEST_CASE("uint32 ns relative: header stamp plus offset, non-finite points skipped", "[point_cloud_read]") {
  const PointCloud2 msg = make_cloud<std::uint32_t>({0, NOT_FINITE, 2}, {300, 999, 100}, PointField::UINT32);
  const auto scan = point_cloud2_to_scan(msg, {.field = "t", .unit = 1ns, .relative = true});
  REQUIRE(scan.points.size() == 2);
  REQUIRE(scan.points[1].x() == 2);
  REQUIRE(scan.timestamps.per_point == TimestampVector{HEADER_STAMP + 300ns, HEADER_STAMP + 100ns});
  REQUIRE(scan.timestamps.min == HEADER_STAMP + 100ns);
  REQUIRE(scan.timestamps.max == HEADER_STAMP + 300ns);
}

TEST_CASE("uint32 us relative", "[point_cloud_read]") {
  const PointCloud2 msg = make_cloud<std::uint32_t>({0, 1}, {1500, 2500}, PointField::UINT32);
  const auto scan = point_cloud2_to_scan(msg, {.field = "t", .unit = 1us, .relative = true});
  REQUIRE(scan.timestamps.per_point == TimestampVector{HEADER_STAMP + 1500us, HEADER_STAMP + 2500us});
}

TEST_CASE("float32 s relative, offsets before the header stamp included", "[point_cloud_read]") {
  const PointCloud2 msg = make_cloud<float>({0, 1}, {0.5F, -0.25F}, PointField::FLOAT32);
  const auto scan = point_cloud2_to_scan(msg, {.field = "t", .unit = 1s, .relative = true});
  REQUIRE(scan.timestamps.per_point == TimestampVector{HEADER_STAMP + 500ms, HEADER_STAMP - 250ms});
  REQUIRE(scan.timestamps.min == HEADER_STAMP - 250ms);
}

TEST_CASE("float64 s absolute: the time itself", "[point_cloud_read]") {
  const PointCloud2 msg = make_cloud<double>({0, 1}, {1'700'000'000.5, 1'700'000'001.0}, PointField::FLOAT64);
  const auto scan = point_cloud2_to_scan(msg, {.field = "t", .unit = 1s, .relative = false});
  REQUIRE(scan.timestamps.per_point ==
          TimestampVector{Nsec{1'700'000'000'500'000'000}, Nsec{1'700'000'001'000'000'000}});
}

TEST_CASE("min and max are the earliest and latest stamp, in any point order", "[point_cloud_read]") {
  const PointCloud2 msg = make_cloud<std::uint32_t>({0, 1, 2}, {200, 100, 300}, PointField::UINT32);
  const auto scan = point_cloud2_to_scan(msg, {.field = "t", .unit = 1ns, .relative = true});
  REQUIRE(scan.points.size() == 3);
  REQUIRE(scan.timestamps.min == HEADER_STAMP + 100ns);
  REQUIRE(scan.timestamps.max == HEADER_STAMP + 300ns);
}

TEST_CASE("empty field: every point at the header stamp", "[point_cloud_read]") {
  const PointCloud2 msg = make_cloud<std::uint32_t>({0, NOT_FINITE, 2}, {200, 100, 300}, PointField::UINT32);
  const auto scan = point_cloud2_to_scan(msg, {.field = "", .unit = 1ns, .relative = true});
  REQUIRE(scan.points.size() == 2);
  REQUIRE(scan.timestamps.min == HEADER_STAMP);
  REQUIRE(scan.timestamps.max == HEADER_STAMP);
}

TEST_CASE("a scan without a finite point throws InputError", "[point_cloud_read]") {
  const PointCloud2 msg = make_cloud<std::uint32_t>({NOT_FINITE, NOT_FINITE}, {0, 1}, PointField::UINT32);
  REQUIRE_THROWS_AS(point_cloud2_to_scan(msg, {.field = "t", .unit = 1ns, .relative = true}),
                    rko_lio::core::InputError);
}

TEST_CASE("a missing or unsupported time field throws invalid_argument", "[point_cloud_read]") {
  PointCloud2 msg = make_cloud<std::uint32_t>({0, 1}, {0, 1}, PointField::UINT32);
  REQUIRE_THROWS_AS(point_cloud2_to_scan(msg, {.field = "time", .unit = 1ns, .relative = true}),
                    std::invalid_argument);
  msg.fields.back().datatype = PointField::INT32;
  REQUIRE_THROWS_AS(point_cloud2_to_scan(msg, {.field = "t", .unit = 1ns, .relative = true}),
                    std::invalid_argument);
}

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

#include "point_cloud_read.hpp"
#include "rko_lio/core/error.hpp"
#include "rosbag.hpp"
#include "time.hpp"
#include <point_cloud_interfaces/msg/compressed_point_cloud2.hpp>
#include <rclcpp/utilities.hpp>
#include <rclcpp/version.h>
#include <spdlog/spdlog.h>
// stl
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace rko_lio::ros::utils {
using point_cloud_interfaces::msg::CompressedPointCloud2;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

namespace {
using sensor_msgs::PointCloud2ConstIterator;

Scan read_scan_without_time(const PointCloud2& msg) {
  const size_t point_count = static_cast<size_t>(msg.height) * msg.width;
  const core::Nsec stamp = to_ns(msg.header.stamp);
  Scan scan{.timestamps = {.min = stamp, .max = stamp}};
  scan.points.reserve(point_count);
  PointCloud2ConstIterator<float> msg_x(msg, "x");
  PointCloud2ConstIterator<float> msg_y(msg, "y");
  PointCloud2ConstIterator<float> msg_z(msg, "z");
  for (size_t i = 0; i < point_count; ++i, ++msg_x, ++msg_y, ++msg_z) {
    if (std::isfinite(*msg_x) && std::isfinite(*msg_y) && std::isfinite(*msg_z)) {
      scan.points.emplace_back(*msg_x, *msg_y, *msg_z);
    }
  }
  if (scan.points.empty()) {
    throw core::InputError("A scan of " + std::to_string(point_count) + " points has no finite point.");
  }
  return scan;
}

template <typename T>
Scan read_scan(const PointCloud2& msg, const LidarTime& lidar_time) {
  const size_t point_count = static_cast<size_t>(msg.height) * msg.width;
  const core::Nsec origin = lidar_time.relative ? to_ns(msg.header.stamp) : core::Nsec{0};
  Scan scan{.timestamps = {.min = core::Nsec::max(), .max = core::Nsec::min()}};
  scan.points.reserve(point_count);
  scan.timestamps.per_point.reserve(point_count);
  PointCloud2ConstIterator<float> msg_x(msg, "x");
  PointCloud2ConstIterator<float> msg_y(msg, "y");
  PointCloud2ConstIterator<float> msg_z(msg, "z");
  PointCloud2ConstIterator<T> msg_time(msg, lidar_time.field);
  for (size_t i = 0; i < point_count; ++i, ++msg_x, ++msg_y, ++msg_z, ++msg_time) {
    if (!std::isfinite(*msg_x) || !std::isfinite(*msg_y) || !std::isfinite(*msg_z)) {
      continue;
    }
    scan.points.emplace_back(*msg_x, *msg_y, *msg_z);
    const core::Nsec offset = std::invoke([&] {
      if constexpr (std::is_integral_v<T>) {
        return static_cast<std::int64_t>(*msg_time) * lidar_time.unit;
      } else {
        return core::Nsec(std::llround(static_cast<double>(*msg_time) * static_cast<double>(lidar_time.unit.count())));
      }
    });
    const core::Nsec stamp = origin + offset;
    scan.timestamps.min = std::min(scan.timestamps.min, stamp);
    scan.timestamps.max = std::max(scan.timestamps.max, stamp);
    scan.timestamps.per_point.push_back(stamp);
  }
  if (scan.points.empty()) {
    throw core::InputError("A scan of " + std::to_string(point_count) + " points has no finite point.");
  }
  return scan;
}
} // namespace

Scan point_cloud2_to_scan(const PointCloud2& msg, const LidarTime& lidar_time) {
  if (lidar_time.field.empty()) {
    return read_scan_without_time(msg);
  }
  const auto field = std::ranges::find(msg.fields, lidar_time.field, &PointField::name);
  if (field == msg.fields.end()) {
    std::string field_names;
    for (const PointField& msg_field : msg.fields) {
      field_names += " " + msg_field.name;
    }
    throw std::invalid_argument(
        "lidar_time.field '" + lidar_time.field +
        "' is not a field of the lidar's PointCloud2 (its fields:" + field_names +
        "). Set lidar_time.field to its per-point time field, or, with deskew:=false and no time field, to \"\".");
  }
  switch (field->datatype) {
  case PointField::UINT32:
    return read_scan<std::uint32_t>(msg, lidar_time);
  case PointField::FLOAT32:
    return read_scan<float>(msg, lidar_time);
  case PointField::FLOAT64:
    return read_scan<double>(msg, lidar_time);
  default:
    throw std::invalid_argument("lidar_time.field '" + lidar_time.field + "' has PointField datatype " +
                                std::to_string(field->datatype) + "; supported are uint32, float32 and float64.");
  }
}

LidarDeserializer::LidarDeserializer(const std::string_view type) {
  if (type == rosidl_generator_traits::name<CompressedPointCloud2>()) {
    codec.emplace();
  }
}

PointCloud2::ConstSharedPtr LidarDeserializer::operator()(const std::shared_ptr<rclcpp::SerializedMessage>& msg) const {
  if (!codec) {
    return deserialize<PointCloud2>(*msg);
  }
  if (!decoder) {
    const std::string format = deserialize<CompressedPointCloud2>(*msg)->format;
    decoder = codec->getDecoderByName(format);
    if (!decoder) {
      throw core::InputError("No point_cloud_transport plugin named '" + format + "' is installed.");
    }
  }
  const auto decoded = decoder->decode(msg);
  if (!decoded) {
    spdlog::warn("Dropping scan: could not decode it: {}", decoded.error());
    return nullptr;
  }
  return decoded->value_or(nullptr);
}

rclcpp::SubscriptionBase::SharedPtr
create_lidar_subscription(const rclcpp::Node::SharedPtr& node,
                          const std::string& topic,
                          const rclcpp::QoS& qos,
                          const std::function<void(const PointCloud2::ConstSharedPtr&)>& callback) {
  auto publishers = node->get_publishers_info_by_topic(topic);
  if (publishers.empty()) {
    spdlog::info("Waiting for a publisher on {}.", topic);
  }
  while (publishers.empty()) {
    if (!rclcpp::sleep_for(std::chrono::milliseconds(100), node->get_node_base_interface()->get_context())) {
      return nullptr;
    }
    publishers = node->get_publishers_info_by_topic(topic);
  }
  const std::string& type = publishers.front().topic_type();
  if (type == rosidl_generator_traits::name<CompressedPointCloud2>()) {
    return node->create_generic_subscription(
        topic, type, qos,
#if RCLCPP_VERSION_MAJOR >= 28
        [deserialize_lidar = LidarDeserializer(type), callback](std::unique_ptr<rclcpp::SerializedMessage> msg) {
          if (const auto cloud = deserialize_lidar(std::move(msg))) {
#else
        [deserialize_lidar = LidarDeserializer(type), callback](std::shared_ptr<rclcpp::SerializedMessage> msg) {
          if (const auto cloud = deserialize_lidar(msg)) {
#endif
            callback(cloud);
          }
        });
  }
  return node->create_subscription<PointCloud2>(topic, qos, callback);
}

} // namespace rko_lio::ros::utils

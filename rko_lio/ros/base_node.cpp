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

#include "base_node.hpp"
#include "rko_lio/core/error.hpp"
#include "rko_lio/ros/utils/point_cloud_write.hpp"
#include "rko_lio/ros/utils/ros_vectors.hpp"
#include "rko_lio/ros/utils/spdlog_sink.hpp"
#include "rko_lio/ros/utils/time.hpp"
#include "rko_lio/ros/utils/transforms.hpp"
#include <UTL/profiler.hpp>
// other
#include <algorithm>
#include <cmath>
#include <format>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <rclcpp/utilities.hpp>
#include <rclcpp/version.h>
#include <set>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <string_view>

namespace rko_lio::ros {
namespace {
using namespace std::literals;

// while no payload waits for them, the IMU buffer keeps this much before its newest sample
constexpr core::Nsec MAX_IMU_BUFFER_SPAN = 10s;

// An IMU is stale when its newest sample is more than this older than the newest sample of any IMU. Payloads do not
// wait for a stale IMU.
constexpr core::Nsec MAX_IMU_LAG = 250ms;
// Online, a payload can register before every IMU that is not stale has a sample past its end, once some IMU has one
// and at least this many samples, over all IMUs, are stamped between the previous payload's end and this payload's end.
constexpr std::size_t MIN_IMU_SAMPLES = 10;
// Every LATE_IMU_LOG_PERIOD, warn about each IMU for which more than MAX_LATE_IMU_FRACTION of its samples in that
// period were dropped as late.
constexpr std::chrono::seconds LATE_IMU_LOG_PERIOD = 60s;
constexpr double MAX_LATE_IMU_FRACTION = 0.1;

bool is_stale(const ImuInput& in, const core::Nsec newest_time) {
  return !in.newest_time || *in.newest_time < newest_time - MAX_IMU_LAG;
}

// Merging assumes phase-locked lidars at 10 Hz: one merge payload's scans end within MAX_PHASE_SPREAD of each other,
// one LIDAR_PERIOD after the previous payload.
constexpr core::Nsec MAX_PHASE_SPREAD = 5ms;
constexpr core::Nsec LIDAR_PERIOD = 100ms;
constexpr int MAX_CONSECUTIVE_UNSYNCED = 10;
// a scan more than this many payloads from the anchor is a time step (PTP), not a phase-locked lidar
constexpr int MAX_PAYLOAD_JUMP = 10;
} // namespace

BaseNode::BaseNode(const std::string& node_name, const rclcpp::NodeOptions& options, const bool wait_for_every_imu)
    : min_imu_samples(wait_for_every_imu ? std::nullopt : std::optional<std::size_t>(MIN_IMU_SAMPLES)) {
  node = rclcpp::Node::make_shared(node_name, options);
  auto logger = std::make_shared<spdlog::logger>("rko_lio", std::make_shared<utils::RclcppSink>(node->get_logger()));
  logger->set_level(spdlog::level::trace);
  spdlog::set_default_logger(std::move(logger));

  const auto imu_topic = node->declare_parameter<std::string>("imu_topic", "");
  // set instead of imu_topic to feed the LIO from every listed IMU
  const auto imu_topics = node->declare_parameter<std::vector<std::string>>("imu_topics", std::vector<std::string>{});
  if (imu_topic.empty() == imu_topics.empty()) {
    throw std::invalid_argument("Set exactly one of imu_topic and imu_topics (several IMUs, all used); imu_topic is '" +
                                imu_topic + "' and imu_topics has " + std::to_string(imu_topics.size()) + " entries.");
  }
  if (std::set<std::string>(imu_topics.begin(), imu_topics.end()).size() != imu_topics.size()) {
    throw std::invalid_argument("imu_topics lists an IMU more than once; list each IMU once.");
  }
  for (const std::string& topic : imu_topics.empty() ? std::vector<std::string>{imu_topic} : imu_topics) {
    imu_inputs.push_back({.topic = topic});
  }
  lidar_topic = node->declare_parameter<std::string>("lidar_topic", lidar_topic);
  lidar_topics = node->declare_parameter<std::vector<std::string>>("lidar_topics", lidar_topics);
  if (lidar_topic.empty() == lidar_topics.empty()) {
    throw std::invalid_argument(
        "Set exactly one of lidar_topic and lidar_topics (several lidars merged into one payload).");
  }
  base_frame = node->declare_parameter<std::string>("base_frame"); // required
  odom_frame = node->declare_parameter<std::string>("odom_frame", odom_frame);
  odom_topic = node->declare_parameter<std::string>("odom_topic", odom_topic);

  // tf
  invert_odom_tf = node->declare_parameter<bool>("invert_odom_tf", invert_odom_tf);
  tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
#if RCLCPP_VERSION_MAJOR >= 30
  tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer, *node);
#else
  tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer, node);
#endif

  // publishing
  const rclcpp::QoS publisher_qos(rclcpp::SystemDefaultsQoS().keep_last(1).durability_volatile());
  publish_odom = node->declare_parameter<bool>("publish_odom", publish_odom);
  publish_odom_tf = node->declare_parameter<bool>("publish_odom_tf", publish_odom_tf);
  if (publish_odom_tf) {
    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*node);
  }
  if (publish_odom) {
    odom_publisher = node->create_publisher<nav_msgs::msg::Odometry>(odom_topic, publisher_qos);
    reset_count_publisher = node->create_publisher<std_msgs::msg::UInt32>(
        "rko_lio/reset_count", rclcpp::SystemDefaultsQoS().keep_last(1).transient_local());
    reset_count_publisher->publish(std_msgs::msg::UInt32());
  }

  publish_deskewed_scan = node->declare_parameter<bool>("publish_deskewed_scan", publish_deskewed_scan);
  if (publish_deskewed_scan) {
    deskewed_scan_topic = node->declare_parameter<std::string>("deskewed_scan_topic", deskewed_scan_topic);
    frame_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(deskewed_scan_topic, publisher_qos);
  }

  publish_local_map = node->declare_parameter<bool>("publish_local_map", publish_local_map);
  if (publish_local_map) {
    map_topic = node->declare_parameter<std::string>("map_topic", map_topic);
    const double publish_map_after_seconds =
        node->declare_parameter<double>("publish_map_after", core::to_seconds(publish_map_after));
    publish_map_after =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(publish_map_after_seconds));
    map_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(map_topic, publisher_qos);
  }

  // lio params
  core::LIO::Config lio_config{};
  lio_config.deskew = node->declare_parameter<bool>("deskew", lio_config.deskew);
  lio_config.deskew_per_stamp = node->declare_parameter<bool>("deskew_per_stamp", lio_config.deskew_per_stamp);
  lio_config.max_iterations = static_cast<std::size_t>(node->declare_parameter<int>("max_iterations", 50));
  lio_config.voxel_size =
      static_cast<core::Scalar>(node->declare_parameter<double>("voxel_size", lio_config.voxel_size));
  lio_config.max_range = static_cast<core::Scalar>(node->declare_parameter<double>("max_range", lio_config.max_range));
  lio_config.min_range = static_cast<core::Scalar>(node->declare_parameter<double>("min_range", lio_config.min_range));
  lio_config.convergence_criterion = static_cast<core::Scalar>(
      node->declare_parameter<double>("convergence_criterion", lio_config.convergence_criterion));
  lio_config.max_correspondence_distance = static_cast<core::Scalar>(
      node->declare_parameter<double>("max_correspondence_distance", lio_config.max_correspondence_distance));
  lio_config.max_num_threads = static_cast<int>(node->declare_parameter<int>("max_num_threads", 1));
  lio_config.initialization_phase =
      node->declare_parameter<bool>("initialization_phase", lio_config.initialization_phase);
  lio_config.double_downsample = node->declare_parameter<bool>("double_downsample", lio_config.double_downsample);
  lio_config.flat = node->declare_parameter<bool>("flat", lio_config.flat);
  lio = std::make_unique<core::LIO>(lio_config);

  lidar_time.field = node->declare_parameter<std::string>("lidar_time.field", "t");
  if (lio->config.deskew && lidar_time.field.empty()) {
    throw std::invalid_argument("lidar_time.field is empty: deskewing needs per-point time. Set it, or deskew:=false.");
  }
  const std::map<std::string, core::Nsec> units{{"ns", 1ns}, {"us", 1us}, {"s", 1s}};
  const std::string unit = node->declare_parameter<std::string>("lidar_time.unit", "ns");
  if (!units.contains(unit)) {
    throw std::invalid_argument("lidar_time.unit is '" + unit + "'; it is one of ns, us and s.");
  }
  lidar_time.unit = units.at(unit);
  const std::string reference = node->declare_parameter<std::string>("lidar_time.reference", "relative");
  if (reference != "relative" && reference != "absolute") {
    throw std::invalid_argument("lidar_time.reference is '" + reference + "'; it is relative or absolute.");
  }
  lidar_time.relative = reference == "relative";

  std::string imus;
  for (const ImuInput& in : imu_inputs) {
    imus += (imus.empty() ? "" : ", ") + in.topic;
  }
  RCLCPP_INFO_STREAM(
      node->get_logger(),
      "IMU: " << imus << ", LiDAR: "
              << (lidar_topics.empty() ? lidar_topic : "a merge of " + std::to_string(lidar_topics.size()) + " lidars")
              << ". Max number of threads: " << lio_config.max_num_threads << ". "
              << (publish_odom ? "Publishing odometry to " + odom_topic + " ( " + odom_frame + " )"
                               : std::string("Not publishing odometry"))
              << (publish_odom_tf ? ", its tf" : ", no tf") << ". Deskewing is "
              << (lio->config.deskew ? "enabled" : "disabled") << "."
              << (publish_deskewed_scan ? (" Publishing the deskewed scan to " + deskewed_scan_topic + ".") : ""));

  reset_on_registration_error =
      node->declare_parameter<bool>("reset_on_registration_error", reset_on_registration_error);
  reset_service = node->create_service<std_srvs::srv::Trigger>(
      "~/reset", [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>&,
                        const std::shared_ptr<std_srvs::srv::Trigger::Response>& response) {
        reset_requested = true;
        response->success = true;
        response->message = "The odometry resets before its next payload.";
      });

  max_lidar_buffer_size = static_cast<std::size_t>(
      node->declare_parameter<int>("max_lidar_buffer_size", static_cast<int>(max_lidar_buffer_size)));
  if (node->declare_parameter<bool>("publish_metrics", false)) {
    metrics_publisher = node->create_publisher<std_msgs::msg::Float64MultiArray>("rko_lio/metrics", 100);
  }
  for (const std::string& topic : lidar_topics) {
    multi_lidar_inputs.push_back({.topic = topic});
  }
  // last: a throw above must not leave a thread running, and they read lio
  if (publish_local_map) {
    map_publish_thread = std::jthread([this](const std::stop_token& stop) { publish_map_loop(stop); });
  }
  registration_thread = std::jthread([this](const std::stop_token& stop) { registration_loop(stop); });
  RCLCPP_INFO(node->get_logger(), "RKO LIO Node is up!");
}

std::optional<Sophus::SE3s> BaseNode::lookup_extrinsic(const std::string& frame) const {
  if (frame.empty()) {
    throw std::invalid_argument("A message header has no frame id: its extrinsic comes from tf.");
  }
  std::optional<Sophus::SE3s> extrinsic = utils::get_transform(tf_buffer, frame, base_frame);
  if (extrinsic) {
    RCLCPP_INFO_STREAM(node->get_logger(),
                       "Extrinsic from " << frame << " to " << base_frame << ": " << extrinsic->log().transpose());
  }
  return extrinsic;
}

LidarPayload BaseNode::lidar_msg_to_payload(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg,
                                            const Sophus::SE3s& lidar2base) const {
  const auto arrival = std::chrono::steady_clock::now();
  const core::Nsec arrival_node_clock = utils::to_ns(node->now());
  UTL_PROFILER_SCOPE("rko_lio: convert message");
  LidarPayload payload{.arrival = arrival};
  utils::Scan scan = utils::point_cloud2_to_scan(*lidar_msg, lidar_time);
  payload.clouds.push_back(std::move(scan.points));
  payload.timestamps.push_back(std::move(scan.timestamps));
  payload.lidar2base.push_back(lidar2base);
  payload.upstream = arrival_node_clock - payload.end();
  payload.ready = std::chrono::steady_clock::now();
  return payload;
}

void BaseNode::publish_odometry() const {
  if (!publish_odom) {
    return;
  }
  const core::State& state = lio->lidar_state;
  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header.stamp = utils::to_ros_time(state.time);
  odom_msg.header.frame_id = odom_frame;
  odom_msg.child_frame_id = base_frame;
  odom_msg.pose.pose = utils::sophus_to_pose(state.pose);
  odom_msg.twist.twist.linear = utils::eigen_to_ros_vector3(state.linear_velocity);
  odom_msg.twist.twist.angular = utils::eigen_to_ros_vector3(state.angular_velocity);
  odom_publisher->publish(odom_msg);
}

void BaseNode::publish_tf() const {
  if (!publish_odom_tf) {
    return;
  }
  const core::State& state = lio->lidar_state;
  geometry_msgs::msg::TransformStamped transform_msg;
  transform_msg.header.stamp = utils::to_ros_time(state.time);
  if (invert_odom_tf) {
    transform_msg.header.frame_id = base_frame;
    transform_msg.child_frame_id = odom_frame;
    transform_msg.transform = utils::sophus_to_transform(state.pose.inverse());
  } else {
    transform_msg.header.frame_id = odom_frame;
    transform_msg.child_frame_id = base_frame;
    transform_msg.transform = utils::sophus_to_transform(state.pose);
  }
  tf_broadcaster->sendTransform(transform_msg);
}

void BaseNode::publish_deskewed_clouds(const std::vector<core::Vector3sVector>& deskewed_clouds) const {
  if (!publish_deskewed_scan) {
    return;
  }
  std_msgs::msg::Header header;
  header.frame_id = base_frame;
  header.stamp = utils::to_ros_time(lio->lidar_state.time);
  core::Vector3sVector points; // one message for all clouds
  for (const core::Vector3sVector& cloud : deskewed_clouds) {
    points.insert(points.end(), cloud.begin(), cloud.end());
  }
  frame_publisher->publish(utils::eigen_to_point_cloud2(points, header));
}

void BaseNode::publish_map_loop(const std::stop_token& stop) {
  std::condition_variable_any period; // nothing notifies it: each wait ends after publish_map_after, or on stop
  std::unique_lock lock(local_map_mutex);
  while (!period.wait_for(lock, stop, publish_map_after, [] { return false; }) && !stop.stop_requested()) {
    if (lio->map.empty()) {
      RCLCPP_WARN_ONCE(node->get_logger(), "Local map publish thread: Local map is empty.");
      continue;
    }
    const core::Vector3sVector map_points = lio->map.points();
    const core::Nsec map_time = lio->lidar_state.time;
    lock.unlock(); // we don't access the local map anymore
    std_msgs::msg::Header map_header;
    map_header.stamp = utils::to_ros_time(map_time);
    map_header.frame_id = odom_frame;
    map_publisher->publish(utils::eigen_to_point_cloud2(map_points, map_header));
    lock.lock();
  }
}

void BaseNode::reset_odometry() {
  const bool forced_init_off = lio->config.initialization_phase;
  {
    const std::scoped_lock lock(local_map_mutex);
    lio->restart();
  }
  ++reset_count;
  RCLCPP_WARN_STREAM(node->get_logger(), "Odometry reset count: " << reset_count);
  if (publish_odom) {
    std_msgs::msg::UInt32 reset_count_msg;
    reset_count_msg.data = static_cast<std::uint32_t>(reset_count);
    reset_count_publisher->publish(reset_count_msg);
  }
  if (forced_init_off) {
    RCLCPP_WARN(node->get_logger(),
                "initialization_phase was enabled; it is off after reset because static start is not guaranteed.");
  }
}

void BaseNode::imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr& imu_msg, const std::size_t input) {
  ImuInput& in = imu_inputs[input];
  if (!in.imu2base) {
    in.imu2base = lookup_extrinsic(imu_msg->header.frame_id);
  }
  if (!in.imu2base) {
    return;
  }
  {
    const std::scoped_lock lock(buffer_mutex);
    in.buffer.push_back({
        .time = utils::to_ns(imu_msg->header.stamp),
        .acceleration = utils::ros_to_eigen_vector3(imu_msg->linear_acceleration),
        .angular_velocity = utils::ros_to_eigen_vector3(imu_msg->angular_velocity),
    });
    in.newest_time = std::max(in.newest_time.value_or(in.buffer.back().time), in.buffer.back().time);
    while (lidar_buffer.empty() && in.buffer.back().time - in.buffer.front().time > MAX_IMU_BUFFER_SPAN) {
      in.buffer.pop_front();
    }
    update_can_register();
  }
  if (can_register) {
    registration_wakeup.notify_one();
  }
}

void BaseNode::lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg) {
  if (!extrinsic_lidar2base) {
    extrinsic_lidar2base = lookup_extrinsic(lidar_msg->header.frame_id);
  }
  if (!extrinsic_lidar2base) {
    return;
  }
  try {
    add_to_lidar_buffer(lidar_msg_to_payload(lidar_msg, *extrinsic_lidar2base));
  } catch (const core::InputError& ex) {
    RCLCPP_ERROR_STREAM(node->get_logger(), "Dropping scan: " << ex.what());
  }
}

void BaseNode::multi_lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg,
                                    const std::size_t input) {
  MultiLidarInput& in = multi_lidar_inputs[input];
  if (!in.lidar2base) {
    in.lidar2base = lookup_extrinsic(lidar_msg->header.frame_id);
  }
  if (!in.lidar2base) {
    return;
  }
  try {
    LidarPayload payload = lidar_msg_to_payload(lidar_msg, *in.lidar2base);
    if (payload_anchor && std::chrono::abs(payload.end() - payload_anchor->end) > MAX_PAYLOAD_JUMP * LIDAR_PERIOD) {
      RCLCPP_WARN_STREAM(node->get_logger(),
                         "A scan from " << in.topic << " ends " << core::to_seconds(payload.end() - payload_anchor->end)
                                        << " s from the merge anchor: re-anchoring the merge on it");
      open_payloads.clear();
      payload_anchor.reset();
      last_merged_payload_index.reset();
    }
    if (!payload_anchor) {
      payload_anchor = PayloadAnchor{.end = payload.end(), .index = 0};
    }
    const std::int64_t payload_index =
        payload_anchor->index +
        std::llround(core::to_seconds(payload.end() - payload_anchor->end) / core::to_seconds(LIDAR_PERIOD));
    const bool merged = last_merged_payload_index && payload_index <= *last_merged_payload_index;
    const auto drop = [&](const std::string_view reason) {
      RCLCPP_WARN_STREAM(node->get_logger(), "Dropping a LiDAR scan of " << payload.point_count() << " points from "
                                                                         << in.topic << ": " << reason);
      publish_metrics(payload, std::chrono::steady_clock::now(), PayloadStatus::skipped,
                      {.input_points = payload.point_count()});
    };
    if (in.fragment_check.is_fragment(core::to_seconds<core::Scalar>(payload.end() - payload.start()),
                                      static_cast<core::Scalar>(payload.point_count()))) {
      drop("a fragment");
      if (!merged) {
        open_payloads.try_emplace(payload_index, multi_lidar_inputs.size()).first->second[input].dropped = true;
        merge_ready_payloads();
      }
      return;
    }
    if (merged) {
      drop("its payload was merged without it");
      return;
    }
    std::vector<PayloadSlot>& slots = open_payloads.try_emplace(payload_index, multi_lidar_inputs.size()).first->second;
    if (slots[input].payload) {
      drop("a second one in one payload");
      return;
    }
    if (payload_index > payload_anchor->index) {
      payload_anchor = PayloadAnchor{.end = payload.end(), .index = payload_index};
    }
    slots[input].payload = std::move(payload);
    merge_ready_payloads();
  } catch (const core::InputError& ex) {
    RCLCPP_ERROR_STREAM(node->get_logger(), "Dropping scan: " << ex.what());
  }
}

void BaseNode::merge_ready_payloads() {
  // oldest first: complete (every lidar in or dropped), a newer one complete, or one two payloads newer open
  const auto complete = [](const auto& open_payload) {
    return std::ranges::all_of(open_payload.second,
                               [](const PayloadSlot& slot) { return slot.payload || slot.dropped; });
  };
  while (!open_payloads.empty()) {
    const auto oldest = open_payloads.begin();
    const bool newer_complete = std::any_of(std::next(oldest), open_payloads.end(), complete);
    const bool stale = open_payloads.rbegin()->first - oldest->first >= 2;
    if (!complete(*oldest) && !newer_complete && !stale) {
      return;
    }
    last_merged_payload_index = oldest->first;
    std::vector<PayloadSlot> slots = std::move(oldest->second);
    open_payloads.erase(oldest);
    UTL_PROFILER_SCOPE("rko_lio: merge payload");
    LidarPayload merged;
    std::string missing_topics;
    for (std::size_t input = 0; input < slots.size(); ++input) {
      PayloadSlot& slot = slots[input];
      if (!slot.payload) {
        if (!slot.dropped) {
          missing_topics += " " + multi_lidar_inputs[input].topic;
        }
        continue;
      }
      LidarPayload& payload = *slot.payload;
      merged.timestamps.push_back(std::move(payload.timestamps.front()));
      merged.clouds.push_back(std::move(payload.clouds.front()));
      merged.lidar2base.push_back(payload.lidar2base.front());
      if (payload.ready > merged.ready) {
        merged.arrival = payload.arrival;
        merged.ready = payload.ready;
      }
    }
    if (merged.clouds.empty()) {
      continue;
    }
    // upstream includes the wait for the last lidar
    merged.upstream = utils::to_ns(node->now()) - merged.end();
    const core::Nsec first_end = std::ranges::min_element(merged.timestamps, {}, &core::Timestamps::max)->max;
    const core::Nsec last_end = merged.end();
    if (last_end - first_end > MAX_PHASE_SPREAD) {
      const double spread_ms = std::chrono::duration<double, std::milli>(last_end - first_end).count();
      const double max_spread_ms = std::chrono::duration<double, std::milli>(MAX_PHASE_SPREAD).count();
      RCLCPP_ERROR_STREAM(node->get_logger(), "Dropping unsynchronised lidar scans: their ends differ by "
                                                  << spread_ms << " ms, more than the " << max_spread_ms
                                                  << " ms of phase-locked lidars");
      publish_metrics(merged, std::chrono::steady_clock::now(), PayloadStatus::unsynced,
                      {.input_points = merged.point_count()});
      if (++consecutive_unsynced >= MAX_CONSECUTIVE_UNSYNCED) {
        RCLCPP_FATAL(
            node->get_logger(),
            "%d merge payloads in a row unsynchronised: the lidars are not phase-locked and merging them is invalid. "
            "Exiting.",
            consecutive_unsynced);
        throw std::runtime_error("lidar_topics: the lidars are not phase-locked");
      }
      continue;
    }
    consecutive_unsynced = 0;
    if (!missing_topics.empty()) {
      RCLCPP_WARN_STREAM(node->get_logger(), "Merging without a scan in this payload from" << missing_topics);
    }
    add_to_lidar_buffer(std::move(merged));
  }
}

bool imus_cover(const std::span<const ImuInput> inputs,
                const core::Nsec fed_until,
                const core::Nsec end,
                const std::optional<std::size_t> min_samples) {
  const std::optional<core::Nsec> newest_time =
      std::ranges::max_element(inputs, {}, &ImuInput::newest_time)->newest_time;
  if (!newest_time || *newest_time <= end) {
    return false;
  }
  if (std::ranges::all_of(
          inputs, [&](const ImuInput& in) { return is_stale(in, *newest_time) || in.newest_time.value() > end; })) {
    return true;
  }
  if (!min_samples) {
    return false;
  }
  std::size_t samples_before_end = 0;
  for (const ImuInput& in : inputs) {
    samples_before_end += static_cast<std::size_t>(std::ranges::count_if(
        in.buffer, [&](const core::ImuControl& imu) { return fed_until <= imu.time && imu.time < end; }));
  }
  return samples_before_end >= *min_samples;
}

void BaseNode::update_can_register() {
  if (lidar_buffer.empty()) {
    can_register = false;
    return;
  }
  const std::optional<core::Nsec> newest_time =
      std::ranges::max_element(imu_inputs, {}, &ImuInput::newest_time)->newest_time;
  for (ImuInput& in : imu_inputs) {
    const bool stale = newest_time && is_stale(in, *newest_time);
    if (stale && !in.stale) {
      const std::string why =
          in.newest_time
              ? std::format("its newest sample, at {:.3f} s, is more than {} s before the newest IMU sample at "
                            "{:.3f} s",
                            core::to_seconds(*in.newest_time), core::to_seconds(MAX_IMU_LAG),
                            core::to_seconds(*newest_time))
              : std::format("it has no sample yet, and the newest IMU sample is at {:.3f} s",
                            core::to_seconds(*newest_time));
      RCLCPP_WARN_STREAM(node->get_logger(),
                         std::format("IMU {} is stale: {}. Payloads no longer wait for it.", in.topic, why));
      in.stale_from = in.newest_time;
    } else if (!stale && in.stale) {
      const std::string gap = in.stale_from ? std::format("{:.3f} s after its previous one",
                                                          core::to_seconds(in.newest_time.value() - *in.stale_from))
                                            : std::string("its first");
      RCLCPP_INFO_STREAM(node->get_logger(),
                         std::format("IMU {} is back with a sample at {:.3f} s, {}. Payloads wait for it again.",
                                     in.topic, core::to_seconds(in.newest_time.value()), gap));
    }
    in.stale = stale;
  }
  can_register = imus_cover(imu_inputs, fed_until, lidar_buffer.front().end(), min_imu_samples);
}

void BaseNode::feed_imu_before(const core::Nsec end) {
  struct Sample {
    core::ImuControl imu;
    std::size_t input;
  };
  std::vector<Sample> due;
  for (std::size_t input = 0; input < imu_inputs.size(); ++input) {
    for (std::deque<core::ImuControl>& buffer = imu_inputs[input].buffer; !buffer.empty() && buffer.front().time < end;
         buffer.pop_front()) {
      due.push_back({.imu = buffer.front(), .input = input});
    }
  }
  std::ranges::stable_sort(due, {}, [](const Sample& sample) { return sample.imu.time; });
  for (const Sample& sample : due) {
    ImuInput& in = imu_inputs[sample.input];
    if (sample.imu.time < fed_until) {
      ++in.late_samples;
      continue;
    }
    ++in.used_samples;
    lio->add_imu_measurement(in.imu2base.value(), sample.imu, sample.input);
  }
  fed_until = end;

  const auto now = std::chrono::steady_clock::now();
  if (now - late_imu_samples_logged < LATE_IMU_LOG_PERIOD) {
    return;
  }
  for (ImuInput& in : imu_inputs) {
    const std::size_t samples = in.late_samples + in.used_samples;
    if (samples > 0 && static_cast<double>(in.late_samples) > MAX_LATE_IMU_FRACTION * static_cast<double>(samples)) {
      RCLCPP_WARN_STREAM(
          node->get_logger(),
          std::format("IMU {}: {} of {} samples in the last {} s arrived too late for their payload and were dropped",
                      in.topic, in.late_samples, samples, LATE_IMU_LOG_PERIOD.count()));
    }
    in.late_samples = 0;
    in.used_samples = 0;
  }
  late_imu_samples_logged = now;
}

void BaseNode::add_to_lidar_buffer(LidarPayload payload) {
  std::optional<LidarPayload> dropped;
  {
    const std::scoped_lock lock(buffer_mutex);
    if (lidar_buffer.size() >= max_lidar_buffer_size) {
      RCLCPP_WARN_STREAM(node->get_logger(),
                         "Registration lidar buffer limit reached. Dropping oldest buffered payload.");
      dropped = std::move(lidar_buffer.front());
      lidar_buffer.pop();
    }
    lidar_buffer.push(std::move(payload));
    update_can_register();
  }
  if (can_register) {
    registration_wakeup.notify_one();
  }
  if (dropped) {
    publish_metrics(*dropped, std::chrono::steady_clock::now(), PayloadStatus::buffer_full,
                    {.input_points = dropped->point_count()});
  }
}

void BaseNode::registration_loop(const std::stop_token& stop) {
  while (rclcpp::ok()) {
    std::unique_lock buffer_lock(buffer_mutex);
    if (!registration_wakeup.wait(buffer_lock, stop,
                                  [this, &stop] { return can_register.load() && !stop.stop_requested(); })) {
      break; // stop requested
    }
    UTL_PROFILER_SCOPE("rko_lio: ROS registration loop");
    LidarPayload payload = std::move(lidar_buffer.front());
    lidar_buffer.pop();
    const auto registration_start = std::chrono::steady_clock::now();
    registration_busy = true;
    const core::Nsec end_stamp = payload.end();
    feed_imu_before(end_stamp);
    // check if there are more messages buffered already
    update_can_register();
    buffer_lock.unlock(); // we dont touch the buffers anymore

    try {
      if (reset_requested.exchange(false)) {
        RCLCPP_WARN(node->get_logger(), "Odometry reset requested through the reset service.");
        reset_odometry(); // takes local_map_mutex itself
      }
      const std::vector<core::Vector3sVector> deskewed_clouds = std::invoke([&] {
        const std::scoped_lock lock(local_map_mutex); // the map publish thread reads the map
        return lio->register_payload(std::move(payload.clouds), payload.timestamps, payload.lidar2base);
      });
      if (deskewed_clouds.empty()) {
        publish_metrics(payload, registration_start, PayloadStatus::skipped, lio->payload_stats);
      } else {
        publish_tf();
        publish_odometry();
        publish_metrics(payload, registration_start, PayloadStatus::ok, lio->payload_stats);
        publish_deskewed_clouds(deskewed_clouds);
      }
    } catch (const core::InputError& ex) {
      RCLCPP_ERROR_STREAM(node->get_logger(), "Dropping payload: " << ex.what());
      publish_metrics(payload, registration_start, PayloadStatus::input_error, lio->payload_stats);
    } catch (const core::RegistrationError& ex) {
      RCLCPP_ERROR_STREAM(node->get_logger(), "Registration failed (" << ex.what() << ").");
      publish_metrics(payload, registration_start, PayloadStatus::registration_error, lio->payload_stats);
      if (reset_on_registration_error) {
        reset_odometry();
      } else {
        RCLCPP_WARN(node->get_logger(), "Set reset_on_registration_error:=true to reset odometry and continue.");
        throw;
      }
    }
    registration_busy = false;
  }
  map_publish_thread.request_stop();
}

void BaseNode::publish_metrics(const LidarPayload& payload,
                               const std::chrono::steady_clock::time_point registration_start,
                               const PayloadStatus status,
                               const core::PayloadStats& stats) const {
  if (!metrics_publisher) {
    return;
  }
  const std::int64_t end_ns = payload.end().count();
  const std::int64_t duration_ns = (payload.end() - payload.start()).count();
  const std::int64_t upstream_ns = payload.upstream.count();
  const std::int64_t convert_ns = core::Nsec(payload.ready - payload.arrival).count();
  const std::int64_t wait_ns = core::Nsec(registration_start - payload.ready).count();
  const std::int64_t registration_ns = core::Nsec(std::chrono::steady_clock::now() - registration_start).count();
  // rko_lio/metrics, in this order: end_ns, duration_ns, upstream_ns, convert_ns, wait_ns, registration_ns, status,
  // input_points, keypoints, imu_samples, icp_iterations, correspondences, icp_cost. Times in ns; wait is for an IMU
  // sample past the payload end, or the previous payload; registration ends with the odometry published. end_ns as a
  // double is rounded to 256 ns
  std_msgs::msg::Float64MultiArray msg;
  msg.data = {
      static_cast<double>(end_ns),
      static_cast<double>(duration_ns),
      static_cast<double>(upstream_ns),
      static_cast<double>(convert_ns),
      static_cast<double>(wait_ns),
      static_cast<double>(registration_ns),
      static_cast<double>(status),
      static_cast<double>(stats.input_points),
      static_cast<double>(stats.keypoints),
      static_cast<double>(stats.imu_samples),
      static_cast<double>(stats.icp_iterations),
      static_cast<double>(stats.correspondences),
      static_cast<double>(stats.icp_cost),
  };
  metrics_publisher->publish(msg);
}

} // namespace rko_lio::ros

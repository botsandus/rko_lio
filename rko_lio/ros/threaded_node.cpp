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

#include "threaded_node.hpp"
#include "rko_lio/core/error.hpp"
#include "rko_lio/ros/utils/utils.hpp"
#include <UTL/profiler.hpp>
// stl
#include <algorithm>
#include <cmath>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using namespace std::literals;

// while no payload waits for them, the IMU buffer keeps this much before its newest sample
constexpr rko_lio::core::Nsec MAX_IMU_BUFFER_SPAN = 10s;

// Merging assumes phase-locked lidars at 10 Hz: one merge payload's scans end within MAX_PHASE_SPREAD of each other,
// one LIDAR_PERIOD after the previous payload.
constexpr rko_lio::core::Nsec MAX_PHASE_SPREAD = 5ms;
constexpr rko_lio::core::Nsec LIDAR_PERIOD = 100ms;
constexpr int MAX_CONSECUTIVE_UNSYNCED = 10;
// a scan more than this many payloads from the anchor is a time step (PTP), not a phase-locked lidar
constexpr int MAX_PAYLOAD_JUMP = 10;
} // namespace

namespace rko_lio::ros {

ThreadedNode::ThreadedNode(const std::string& node_name, const rclcpp::NodeOptions& options)
    : BaseNode(node_name, options) {
  max_lidar_buffer_size = static_cast<size_t>(
      node->declare_parameter<int>("async.max_lidar_buffer_size", static_cast<int>(max_lidar_buffer_size)));
  if (node->declare_parameter<bool>("publish_metrics", false)) {
    metrics_publisher = node->create_publisher<std_msgs::msg::Float64MultiArray>("rko_lio/metrics", 100);
  }
  for (const std::string& topic : lidar_topics) {
    multi_lidar_inputs.push_back({.topic = topic});
  }
  registration_thread = std::jthread([this] { registration_loop(); });
}

void ThreadedNode::imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr& imu_msg) {
  if (!extrinsic_from_tf(extrinsic_imu2base, imu_msg->header.frame_id)) {
    return;
  }
  {
    const std::scoped_lock lock(buffer_mutex);
    imu_buffer.emplace(imu_msg_to_imu_data(*imu_msg));
    while (lidar_buffer.empty() && imu_buffer.back().time - imu_buffer.front().time > MAX_IMU_BUFFER_SPAN) {
      imu_buffer.pop();
    }
    atomic_can_process = !lidar_buffer.empty() && imu_buffer.back().time > lidar_buffer.front().end();
  }
  if (atomic_can_process) {
    sync_condition_variable.notify_one();
  }
}

void ThreadedNode::lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg) {
  if (!extrinsic_from_tf(extrinsic_lidar2base, lidar_msg->header.frame_id)) {
    return;
  }
  try {
    add_to_lidar_buffer(process_lidar_msg(lidar_msg, *extrinsic_lidar2base));
  } catch (const core::InputError& ex) {
    RCLCPP_ERROR_STREAM(node->get_logger(), "Dropping scan: " << ex.what());
  }
}

void ThreadedNode::multi_lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg,
                                        const std::size_t input) {
  MultiLidarInput& in = multi_lidar_inputs[input];
  if (!extrinsic_from_tf(in.lidar2base, lidar_msg->header.frame_id)) {
    return;
  }
  try {
    LidarPayload scan = process_lidar_msg(lidar_msg, *in.lidar2base);
    if (payload_anchor && std::chrono::abs(scan.end() - payload_anchor->end) > MAX_PAYLOAD_JUMP * LIDAR_PERIOD) {
      RCLCPP_WARN_STREAM(node->get_logger(),
                         "A scan from " << in.topic << " ends " << core::to_seconds(scan.end() - payload_anchor->end)
                                        << " s from the merge anchor: re-anchoring the merge on it");
      open_payloads.clear();
      payload_anchor.reset();
      last_merged_payload_index = std::numeric_limits<std::int64_t>::min();
    }
    if (!payload_anchor) {
      payload_anchor = PayloadAnchor{.end = scan.end(), .index = 0};
    }
    const std::int64_t payload_index =
        payload_anchor->index +
        std::llround(core::to_seconds(scan.end() - payload_anchor->end) / core::to_seconds(LIDAR_PERIOD));
    const auto drop = [&](const std::string_view reason) {
      RCLCPP_WARN_STREAM(node->get_logger(), "Dropping a LiDAR scan of " << scan.point_count() << " points from "
                                                                         << in.topic << ": " << reason);
      publish_metrics(scan, std::chrono::steady_clock::now(), PayloadStatus::skipped,
                      {.input_points = scan.point_count()});
    };
    if (in.fragment_check.is_fragment(core::to_seconds<core::Scalar>(scan.end() - scan.start()),
                                      static_cast<core::Scalar>(scan.point_count()))) {
      drop("a fragment");
      if (payload_index > last_merged_payload_index) {
        open_payloads.try_emplace(payload_index, multi_lidar_inputs.size()).first->second[input].dropped = true;
        merge_ready_payloads();
      }
      return;
    }
    if (payload_index <= last_merged_payload_index) {
      drop("its payload was merged without it");
      return;
    }
    std::vector<PayloadSlot>& slots = open_payloads.try_emplace(payload_index, multi_lidar_inputs.size()).first->second;
    if (slots[input].scan) {
      drop("a second one in one payload");
      return;
    }
    if (payload_index > payload_anchor->index) {
      payload_anchor = PayloadAnchor{.end = scan.end(), .index = payload_index};
    }
    slots[input].scan = std::move(scan);
    merge_ready_payloads();
  } catch (const core::InputError& ex) {
    RCLCPP_ERROR_STREAM(node->get_logger(), "Dropping scan: " << ex.what());
  }
}

void ThreadedNode::merge_ready_payloads() {
  // oldest first: complete (every lidar in or dropped), a newer one complete, or one two payloads newer open
  const auto complete = [](const auto& payload) {
    return std::ranges::all_of(payload.second, [](const PayloadSlot& slot) { return slot.scan || slot.dropped; });
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
    core::Nsec first_end = core::Nsec::max();
    core::Nsec last_end = core::Nsec::min();
    std::string missing_topics;
    for (std::size_t input = 0; input < slots.size(); ++input) {
      PayloadSlot& slot = slots[input];
      if (!slot.scan) {
        if (!slot.dropped) {
          missing_topics += " " + multi_lidar_inputs[input].topic;
        }
        continue;
      }
      LidarPayload& scan = *slot.scan;
      first_end = std::min(first_end, scan.end());
      last_end = std::max(last_end, scan.end());
      merged.timestamps.push_back(std::move(scan.timestamps.front()));
      merged.clouds.push_back(std::move(scan.clouds.front()));
      merged.lidar2base.push_back(scan.lidar2base.front());
      if (scan.ready > merged.ready) {
        merged.arrival = scan.arrival;
        merged.ready = scan.ready;
      }
    }
    if (merged.clouds.empty()) {
      continue;
    }
    // upstream includes the wait for the last lidar
    merged.upstream = utils::to_ns(node->now()) - merged.end();
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

void ThreadedNode::add_to_lidar_buffer(LidarPayload payload) {
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
    atomic_can_process = !imu_buffer.empty() && imu_buffer.back().time > lidar_buffer.front().end();
  }
  if (atomic_can_process) {
    sync_condition_variable.notify_one();
  }
  if (dropped) {
    publish_metrics(*dropped, std::chrono::steady_clock::now(), PayloadStatus::buffer_full,
                    {.input_points = dropped->point_count()});
  }
}

void ThreadedNode::registration_loop() {
  while (rclcpp::ok() && atomic_node_running) {
    std::unique_lock buffer_lock(buffer_mutex);
    sync_condition_variable.wait(buffer_lock, [this] { return !atomic_node_running || atomic_can_process; });
    UTL_PROFILER_SCOPE("rko_lio: ROS registration loop");
    if (!atomic_node_running) {
      // node could have been killed after waiting on the cv
      break;
    }
    LidarPayload payload = std::move(lidar_buffer.front());
    lidar_buffer.pop();
    const auto registration_start = std::chrono::steady_clock::now();
    registration_busy = true;
    const core::Nsec end_stamp = payload.end();
    for (; !imu_buffer.empty() && imu_buffer.front().time < end_stamp; imu_buffer.pop()) {
      const core::ImuControl& imu_data = imu_buffer.front();
      lio->add_imu_measurement(*extrinsic_imu2base, imu_data);
    }
    // check if there are more messages buffered already
    atomic_can_process =
        !imu_buffer.empty() && !lidar_buffer.empty() && imu_buffer.back().time > lidar_buffer.front().end();
    buffer_lock.unlock(); // we dont touch the buffers anymore

    try {
      const std::vector<core::Vector3sVector> deskewed_clouds =
          register_payload_locked(std::move(payload.clouds), payload.timestamps, payload.lidar2base);
      if (deskewed_clouds.empty()) {
        publish_metrics(payload, registration_start, PayloadStatus::skipped, lio->payload_stats);
      } else {
        publish_tf(lio->lidar_state);
        publish_odometry(lio->lidar_state, odom_publisher);
        publish_metrics(payload, registration_start, PayloadStatus::ok, lio->payload_stats);
        publish_lidar_outputs(deskewed_clouds);
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
  atomic_node_running = false;
}

void ThreadedNode::publish_metrics(const LidarPayload& payload,
                                   const std::chrono::steady_clock::time_point registration_start,
                                   const PayloadStatus status,
                                   const core::PayloadStats& stats) {
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

ThreadedNode::~ThreadedNode() {
  atomic_node_running = false;
  sync_condition_variable.notify_all();
}

} // namespace rko_lio::ros

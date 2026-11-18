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
#include <array>
#include <cmath>
#include <format>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using namespace std::literals;

// while no scan waits for them, the IMU buffer keeps this much before its newest sample
constexpr rko_lio::core::Nsec MAX_IMU_BUFFER_SPAN = 10s;

// Merging assumes phase-locked lidars at 10 Hz: one revolution's scans end within MAX_PHASE_SPREAD of each other
// (measured within 0.3 ms on a B18), one LIDAR_PERIOD after the previous revolution.
constexpr rko_lio::core::Nsec MAX_PHASE_SPREAD = 5ms;
constexpr rko_lio::core::Nsec LIDAR_PERIOD = 100ms;
constexpr int MAX_CONSECUTIVE_UNSYNCED = 10;

constexpr std::array<std::string_view, 6> SCAN_STATUS_NAMES = {
    "ok", "skipped", "buffer_full", "input_error", "registration_error", "unsynced"};

std::int64_t ns_between(const std::chrono::steady_clock::time_point from,
                        const std::chrono::steady_clock::time_point to) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count();
}
} // namespace

namespace rko_lio::ros {

ThreadedNode::ThreadedNode(const std::string& node_name, const rclcpp::NodeOptions& options)
    : BaseNode(node_name, options) {
  max_lidar_buffer_size = static_cast<size_t>(
      node->declare_parameter<int>("async.max_lidar_buffer_size", static_cast<int>(max_lidar_buffer_size)));
  if (node->declare_parameter<bool>("publish_metrics", false)) {
    scan_metrics_publisher = node->create_publisher<std_msgs::msg::Float64MultiArray>("rko_lio/scan_metrics", 100);
  }
  for (const std::string& topic : lidar_topics) {
    merge_inputs.push_back({.topic = topic});
  }
  registration_thread = std::jthread([this] { registration_loop(); });
}

void ThreadedNode::imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr& imu_msg) {
  if (!ensure_frame_and_extrinsics(imu_frame, imu_msg->header.frame_id, "IMU")) {
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
  const auto arrival = std::chrono::steady_clock::now();
  const core::Nsec arrival_node_clock = utils::to_ns(node->now());
  if (!ensure_frame_and_extrinsics(lidar_frame, lidar_msg->header.frame_id, "LiDAR")) {
    return;
  }
  try {
    LidarScan scan = [&] {
      UTL_PROFILER_SCOPE("rko_lio: convert message");
      return process_lidar_msg(lidar_msg);
    }();
    scan.arrival = arrival;
    scan.transport = arrival_node_clock - scan.end();
    enqueue_scan(std::move(scan));
  } catch (const core::InputError& ex) {
    RCLCPP_ERROR_STREAM(node->get_logger(), "Dropping scan: " << ex.what());
  }
}

void ThreadedNode::merge_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg,
                                  const std::size_t input) {
  const auto arrival = std::chrono::steady_clock::now();
  const core::Nsec arrival_node_clock = utils::to_ns(node->now());
  MergeInput& in = merge_inputs[input];
  if (!in.lidar2base) {
    in.lidar2base = utils::get_transform(tf_buffer, lidar_msg->header.frame_id, base_frame, 0s);
    if (!in.lidar2base) {
      return;
    }
    RCLCPP_INFO_STREAM(node->get_logger(), "Merging " << in.topic << " (frame " << lidar_msg->header.frame_id << ")");
  }
  if (!check_and_set_extrinsics()) { // the IMU's
    return;
  }
  try {
    LidarScan scan = [&] {
      UTL_PROFILER_SCOPE("rko_lio: convert message");
      return process_lidar_msg(lidar_msg);
    }();
    scan.arrival = arrival;
    scan.transport = arrival_node_clock - scan.end();
    if (in.fragments.is_fragment(core::to_seconds<core::Scalar>(scan.end() - scan.start()),
                                 static_cast<core::Scalar>(scan.point_count()))) {
      RCLCPP_WARN_STREAM(node->get_logger(),
                         "Dropping a LiDAR scan fragment of " << scan.point_count() << " points from " << in.topic);
      record_scan(scan, std::chrono::steady_clock::now(), ScanStatus::skipped, {.input_points = scan.point_count()});
      return;
    }
    {
      UTL_PROFILER_SCOPE("rko_lio: merge, to base frame");
      for (Eigen::Vector3s& point : scan.points.front()) {
        point = *in.lidar2base * point;
      }
    }
    if (!revolution_origin) {
      revolution_origin = scan.end();
    }
    const std::int64_t revolution =
        std::llround(core::to_seconds(scan.end() - *revolution_origin) / core::to_seconds(LIDAR_PERIOD));
    std::vector<std::optional<LidarScan>>& slots = open_revolutions[revolution];
    slots.resize(merge_inputs.size());
    if (revolution <= last_merged_revolution || slots[input]) {
      RCLCPP_WARN_STREAM(node->get_logger(), "Dropping a LiDAR scan from "
                                                 << in.topic
                                                 << (slots[input] ? ": a second one in one revolution"
                                                                  : ": its revolution was merged without it"));
      record_scan(scan, std::chrono::steady_clock::now(), ScanStatus::skipped, {.input_points = scan.point_count()});
      if (revolution <= last_merged_revolution) {
        open_revolutions.erase(revolution);
      }
      return;
    }
    slots[input] = std::move(scan);
    merge_ready_revolutions();
  } catch (const core::InputError& ex) {
    RCLCPP_ERROR_STREAM(node->get_logger(), "Dropping scan: " << ex.what());
  }
}

void ThreadedNode::merge_ready_revolutions() {
  // oldest first: a complete revolution; an incomplete one once a newer one is complete or two newer ones are open
  // (its missing lidars dropped that scan)
  const auto complete = [](const auto& revolution) {
    return std::ranges::all_of(revolution.second, [](const auto& slot) { return slot.has_value(); });
  };
  while (!open_revolutions.empty()) {
    const auto oldest = open_revolutions.begin();
    const bool newer_complete = std::any_of(std::next(oldest), open_revolutions.end(), complete);
    const bool stale = open_revolutions.rbegin()->first - oldest->first >= 2;
    if (!complete(*oldest) && !newer_complete && !stale) {
      return;
    }
    last_merged_revolution = oldest->first;
    merge_revolution(oldest->second);
    open_revolutions.erase(oldest);
  }
}

void ThreadedNode::merge_revolution(std::vector<std::optional<LidarScan>>& slots) {
  UTL_PROFILER_SCOPE("rko_lio: merge revolution");
  LidarScan merged;
  core::Nsec start = core::Nsec::max();
  core::Nsec first_end = core::Nsec::max();
  core::Nsec last_end = core::Nsec::min();
  std::string missing;
  for (std::size_t input = 0; input < slots.size(); ++input) {
    if (!slots[input]) {
      missing += " " + merge_inputs[input].topic;
      continue;
    }
    LidarScan& scan = *slots[input];
    start = std::min(start, scan.start());
    first_end = std::min(first_end, scan.end());
    last_end = std::max(last_end, scan.end());
    merged.timestamps.push_back(std::move(scan.timestamps.front()));
    merged.points.push_back(std::move(scan.points.front()));
  }
  for (core::Timestamps& timestamps : merged.timestamps) {
    timestamps.min = start;
    timestamps.max = last_end;
  }
  // from the scan end until the merge exists, the wait for every lidar included; queue then holds only the backlog
  merged.arrival = std::chrono::steady_clock::now();
  merged.transport = utils::to_ns(node->now()) - merged.end();
  if (last_end - first_end > MAX_PHASE_SPREAD) {
    const double spread_ms = std::chrono::duration<double, std::milli>(last_end - first_end).count();
    const double max_spread_ms = std::chrono::duration<double, std::milli>(MAX_PHASE_SPREAD).count();
    RCLCPP_ERROR_STREAM(node->get_logger(), "Dropping unsynchronised lidar scans: their ends differ by "
                                                << spread_ms << " ms, more than the " << max_spread_ms
                                                << " ms of phase-locked lidars");
    record_scan(merged, std::chrono::steady_clock::now(), ScanStatus::unsynced, {.input_points = merged.point_count()});
    if (++consecutive_unsynced >= MAX_CONSECUTIVE_UNSYNCED) {
      RCLCPP_FATAL(
          node->get_logger(),
          "%d revolutions in a row unsynchronised: the lidars are not phase-locked and merging them is invalid. "
          "Exiting.",
          consecutive_unsynced);
      stop_results_writer();
      throw std::runtime_error("lidar_topics: the lidars are not phase-locked");
    }
    return;
  }
  consecutive_unsynced = 0;
  if (!missing.empty()) {
    RCLCPP_WARN_STREAM(node->get_logger(), "Merging without a scan this revolution from" << missing);
  }
  enqueue_scan(std::move(merged));
}

void ThreadedNode::enqueue_scan(LidarScan scan) {
  std::optional<LidarScan> dropped;
  {
    const std::scoped_lock lock(buffer_mutex);
    if (lidar_buffer.size() >= max_lidar_buffer_size) {
      RCLCPP_WARN_STREAM(node->get_logger(),
                         "Registration lidar buffer limit reached. Dropping oldest buffered frame.");
      dropped = std::move(lidar_buffer.front());
      lidar_buffer.pop();
    }
    lidar_buffer.push(std::move(scan));
    atomic_can_process = !imu_buffer.empty() && imu_buffer.back().time > lidar_buffer.front().end();
  }
  if (atomic_can_process) {
    sync_condition_variable.notify_one();
  }
  if (dropped) {
    record_scan(*dropped, std::chrono::steady_clock::now(), ScanStatus::buffer_full, {});
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
    LidarScan scan = std::move(lidar_buffer.front());
    lidar_buffer.pop();
    const auto process_start = std::chrono::steady_clock::now();
    registration_busy = true;
    const core::Nsec end_stamp = scan.end();
    for (; !imu_buffer.empty() && imu_buffer.front().time < end_stamp; imu_buffer.pop()) {
      const core::ImuControl& imu_data = imu_buffer.front();
      lio->add_imu_measurement(extrinsic_imu2base, imu_data);
    }
    // check if there are more messages buffered already
    atomic_can_process =
        !imu_buffer.empty() && !lidar_buffer.empty() && imu_buffer.back().time > lidar_buffer.front().end();
    buffer_lock.unlock(); // we dont touch the buffers anymore

    try {
      const std::vector<core::Vector3sVector> deskewed_scan =
          register_scan_locked(std::move(scan.points), scan.timestamps);
      if (!deskewed_scan.empty()) {
        // TODO: first scan is skipped and an empty scan is returned. improve how we handle this
        publish_tf(lio->lidar_state);
        publish_lidar_outputs(deskewed_scan);
      }
      queue_poses_for_results();
      record_scan(scan, process_start, deskewed_scan.empty() ? ScanStatus::skipped : ScanStatus::ok, lio->scan_stats);
    } catch (const core::InputError& ex) {
      RCLCPP_ERROR_STREAM(node->get_logger(), "Dropping scan: " << ex.what());
      record_scan(scan, process_start, ScanStatus::input_error, lio->scan_stats);
    } catch (const core::RegistrationError& ex) {
      RCLCPP_ERROR_STREAM(node->get_logger(), "Registration failed (" << ex.what() << ").");
      record_scan(scan, process_start, ScanStatus::registration_error, lio->scan_stats);
      if (reset_on_registration_error) {
        reset_odometry();
      } else {
        RCLCPP_WARN(node->get_logger(), "Set reset_on_registration_error:=true to reset odometry and continue.");
        stop_results_writer(); // the rethrow terminates without unwinding
        throw;
      }
    }
    registration_busy = false;
  }
  atomic_node_running = false;
}

void ThreadedNode::record_scan(const LidarScan& scan,
                               const std::chrono::steady_clock::time_point process_start,
                               const ScanStatus status,
                               const core::ScanStats& stats) {
  if (!scan_metrics_publisher && !dump_results) {
    return;
  }
  const std::int64_t scan_end = scan.end().count();
  const std::int64_t duration = (scan.end() - scan.start()).count();
  const std::int64_t transport = scan.transport.count();
  const std::int64_t queue = ns_between(scan.arrival, process_start);
  const std::int64_t process = ns_between(process_start, std::chrono::steady_clock::now());
  if (scan_metrics_publisher) { // scan_end_ns as a double is rounded to 256 ns; the other values are exact
    std_msgs::msg::Float64MultiArray msg;
    msg.layout.dim.resize(1);
    msg.layout.dim[0].label = std::string(SCAN_STATISTICS_FIELDS);
    msg.data = {static_cast<double>(scan_end),
                static_cast<double>(duration),
                static_cast<double>(transport),
                static_cast<double>(queue),
                static_cast<double>(process),
                static_cast<double>(status),
                static_cast<double>(stats.input_points),
                static_cast<double>(stats.keypoints),
                static_cast<double>(stats.imu_samples),
                static_cast<double>(stats.icp_iterations),
                static_cast<double>(stats.correspondences),
                static_cast<double>(stats.icp_cost)};
    msg.layout.dim[0].size = static_cast<std::uint32_t>(msg.data.size());
    msg.layout.dim[0].stride = msg.layout.dim[0].size;
    scan_metrics_publisher->publish(msg);
  }
  if (!dump_results) {
    return;
  }
  std::string row =
      std::format("{},{},{},{},{},{},{},{},{},{},{},{}\n", scan_end, duration, transport, queue, process,
                  SCAN_STATUS_NAMES[static_cast<std::size_t>(status)], stats.input_points, stats.keypoints,
                  stats.imu_samples, stats.icp_iterations, stats.correspondences, stats.icp_cost);
  const std::scoped_lock lock(results_mutex);
  pending_scans += row;
}

ThreadedNode::~ThreadedNode() {
  atomic_node_running = false;
  sync_condition_variable.notify_all();
}

} // namespace rko_lio::ros

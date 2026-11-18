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
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using namespace std::literals;

// while no scan waits for them, the IMU buffer keeps this much before its newest sample
constexpr rko_lio::core::Nsec MAX_IMU_BUFFER_SPAN = 10s;

// Merging assumes phase-locked lidars: one revolution's scans end within MAX_PHASE_SPREAD of each other (measured
// within 0.25 ms on a B18). A scan ending more than half a 10 Hz period after the open revolution belongs to the next
// one.
constexpr rko_lio::core::Nsec MAX_PHASE_SPREAD = 5ms;
constexpr rko_lio::core::Nsec NEXT_REVOLUTION = 50ms;
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
    atomic_can_process = !lidar_buffer.empty() && imu_buffer.back().time > lidar_buffer.front().timestamps.max;
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
    LidarScan scan = process_lidar_msg(lidar_msg);
    scan.arrival = arrival;
    scan.transport = arrival_node_clock - scan.timestamps.max;
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
    LidarScan scan = process_lidar_msg(lidar_msg);
    scan.arrival = arrival;
    scan.transport = arrival_node_clock - scan.timestamps.max;
    if (in.fragments.is_fragment(core::to_seconds<core::Scalar>(scan.timestamps.max - scan.timestamps.min),
                                 static_cast<core::Scalar>(scan.points.size()))) {
      RCLCPP_WARN_STREAM(node->get_logger(),
                         "Dropping a LiDAR scan fragment of " << scan.points.size() << " points from " << in.topic);
      record_scan(scan, std::chrono::steady_clock::now(), ScanStatus::skipped, {.input_points = scan.points.size()});
      return;
    }
    for (Eigen::Vector3s& point : scan.points) {
      point = *in.lidar2base * point;
    }
    const auto open = std::ranges::find_if(merge_inputs, [](const MergeInput& m) { return m.scan.has_value(); });
    if (in.scan || (open != merge_inputs.end() &&
                    std::chrono::abs(scan.timestamps.max - open->scan->timestamps.max) > NEXT_REVOLUTION)) {
      merge_open_revolution(); // a lidar missed the open revolution: merge what came
    }
    in.scan = std::move(scan);
    if (std::ranges::all_of(merge_inputs, [](const MergeInput& m) { return m.scan.has_value(); })) {
      merge_open_revolution();
    }
  } catch (const core::InputError& ex) {
    RCLCPP_ERROR_STREAM(node->get_logger(), "Dropping scan: " << ex.what());
  }
}

void ThreadedNode::merge_open_revolution() {
  LidarScan merged{.timestamps = {.min = core::Nsec::max(), .max = core::Nsec::min()}};
  core::Nsec first_end = core::Nsec::max();
  core::Nsec last_end = core::Nsec::min();
  core::Nsec last_arrival_node_clock = core::Nsec::min();
  std::string missing;
  for (MergeInput& m : merge_inputs) {
    if (!m.scan) {
      missing += " " + m.topic;
      continue;
    }
    LidarScan& scan = *m.scan;
    merged.timestamps.min = std::min(merged.timestamps.min, scan.timestamps.min);
    merged.timestamps.max = std::max(merged.timestamps.max, scan.timestamps.max);
    merged.timestamps.per_point.insert(merged.timestamps.per_point.end(), scan.timestamps.per_point.begin(),
                                       scan.timestamps.per_point.end());
    merged.points.insert(merged.points.end(), scan.points.begin(), scan.points.end());
    merged.arrival = std::max(merged.arrival, scan.arrival);
    first_end = std::min(first_end, scan.timestamps.max);
    last_end = std::max(last_end, scan.timestamps.max);
    last_arrival_node_clock = std::max(last_arrival_node_clock, scan.timestamps.max + scan.transport);
    m.scan.reset();
  }
  merged.transport = last_arrival_node_clock - merged.timestamps.max;
  if (last_end - first_end > MAX_PHASE_SPREAD) {
    const double spread_ms = std::chrono::duration<double, std::milli>(last_end - first_end).count();
    const double max_spread_ms = std::chrono::duration<double, std::milli>(MAX_PHASE_SPREAD).count();
    RCLCPP_ERROR_STREAM(node->get_logger(), "Dropping unsynchronised lidar scans: their ends differ by "
                                                << spread_ms << " ms, more than the " << max_spread_ms
                                                << " ms of phase-locked lidars");
    record_scan(merged, std::chrono::steady_clock::now(), ScanStatus::unsynced, {.input_points = merged.points.size()});
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
    atomic_can_process = !imu_buffer.empty() && imu_buffer.back().time > lidar_buffer.front().timestamps.max;
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
    const core::Nsec end_stamp = scan.timestamps.max;
    for (; !imu_buffer.empty() && imu_buffer.front().time < end_stamp; imu_buffer.pop()) {
      const core::ImuControl& imu_data = imu_buffer.front();
      lio->add_imu_measurement(extrinsic_imu2base, imu_data);
    }
    // check if there are more messages buffered already
    atomic_can_process =
        !imu_buffer.empty() && !lidar_buffer.empty() && imu_buffer.back().time > lidar_buffer.front().timestamps.max;
    buffer_lock.unlock(); // we dont touch the buffers anymore

    try {
      const core::Vector3sVector deskewed_scan = register_scan_locked(std::move(scan.points), scan.timestamps);
      if (!deskewed_scan.empty()) {
        // TODO: first scan is skipped and an empty scan is returned. improve how we handle this
        publish_tf(lio->lidar_state);
        publish_lidar_outputs(deskewed_scan);
      }
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
  const std::int64_t scan_end = scan.timestamps.max.count();
  const std::int64_t duration = (scan.timestamps.max - scan.timestamps.min).count();
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

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
#include <array>
#include <filesystem>
#include <format>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace {
using namespace std::literals;

// A scan's row, in the file's header and the published message's layout label. Times in ns: the scan's last point
// (sensor clock) and duration; transport: its arrival on the node clock minus its last point; queue and process: the
// steady clock from arrival to the start of its processing, and the processing.
constexpr std::string_view SCAN_METRICS_FIELDS = "scan_end_ns,scan_duration_ns,transport_ns,queue_ns,process_ns,status,"
                                                 "input_points,keypoints,imu_samples,icp_iterations,correspondences,"
                                                 "icp_cost";
constexpr std::array<std::string_view, 5> SCAN_STATUS_NAMES = {"ok", "skipped", "buffer_full", "input_error",
                                                               "registration_error"};

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
  metrics_dir = node->declare_parameter<std::string>("metrics_dir", metrics_dir);
  if (!metrics_dir.empty()) {
    write_interval = std::chrono::seconds(
        node->declare_parameter<int>("write_interval_s", static_cast<int>(write_interval.count())));
    if (write_interval < 1s) {
      throw std::invalid_argument("write_interval_s must be at least 1 s.");
    }
    std::filesystem::create_directories(metrics_dir);
    const std::string base = (std::filesystem::path(metrics_dir) /
                              std::format("{}_{:%Y%m%dT%H%M%SZ}", node->get_name(),
                                          std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now())))
                                 .string();
    poses_file.open(base + "_poses.tum", std::ios::app);
    scans_file.open(base + "_scans.csv", std::ios::app);
    if (!poses_file || !scans_file) {
      throw std::runtime_error("Cannot open the metrics files " + base + "_{poses.tum,scans.csv}");
    }
    scans_file << SCAN_METRICS_FIELDS << "\n" << std::flush;
    metrics_write_thread = std::jthread([this](const std::stop_token& stop) { write_metrics_loop(stop); });
    RCLCPP_INFO_STREAM(node->get_logger(), "Writing poses and scan statistics to " << base << "_* every "
                                                                                   << write_interval.count() << " s");
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
  } catch (const core::InputError& ex) {
    RCLCPP_ERROR_STREAM(node->get_logger(), "Dropping scan: " << ex.what());
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
  if (!scan_metrics_publisher && metrics_dir.empty()) {
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
    msg.layout.dim[0].label = std::string(SCAN_METRICS_FIELDS);
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
  if (metrics_dir.empty()) {
    return;
  }
  std::string row =
      std::format("{},{},{},{},{},{},{},{},{},{},{},{}\n", scan_end, duration, transport, queue, process,
                  SCAN_STATUS_NAMES[static_cast<std::size_t>(status)], stats.input_points, stats.keypoints,
                  stats.imu_samples, stats.icp_iterations, stats.correspondences, stats.icp_cost);
  std::string pose;
  if (status == ScanStatus::ok) {
    const core::Nsec time = lio->lidar_state.time;
    const Eigen::Vector3s& t = lio->lidar_state.pose.translation();
    const Eigen::Quaternions& q = lio->lidar_state.pose.so3().unit_quaternion();
    pose = std::format("{}.{:09} {} {} {} {} {} {} {}\n", time.count() / 1'000'000'000, time.count() % 1'000'000'000,
                       t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w());
  }
  const std::scoped_lock lock(metrics_mutex);
  pending_scans += row;
  pending_poses += pose;
}

void ThreadedNode::write_metrics_loop(const std::stop_token& stop) {
  std::condition_variable_any wakeup; // only the stop request wakes it early
  bool failed = false;
  while (true) {
    std::string poses;
    std::string scans;
    {
      std::unique_lock lock(metrics_mutex);
      wakeup.wait_for(lock, stop, write_interval, [] { return false; });
      poses.swap(pending_poses);
      scans.swap(pending_scans);
    }
    if (!failed) {
      poses_file << poses << std::flush;
      scans_file << scans << std::flush;
      failed = !poses_file || !scans_file;
      if (failed) {
        RCLCPP_ERROR(node->get_logger(), "Writing the metrics files failed; nothing more is written.");
      }
    }
    if (stop.stop_requested()) {
      return;
    }
  }
}

ThreadedNode::~ThreadedNode() {
  atomic_node_running = false;
  sync_condition_variable.notify_all();
}

} // namespace rko_lio::ros

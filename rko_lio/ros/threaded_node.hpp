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

#pragma once
#include "base_node.hpp"
// stl
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <queue>
#include <stop_token>
#include <string>
#include <thread>
// ros
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

namespace rko_lio::ros {

// What became of a scan: registered, skipped by the odometry (out of order, or a fragment), or dropped.
enum class ScanStatus : std::uint8_t { ok, skipped, buffer_full, input_error, registration_error };

class ThreadedNode : public BaseNode {
public:
  std::mutex buffer_mutex;
  std::condition_variable sync_condition_variable;
  std::atomic<bool> atomic_can_process = false;
  std::atomic<bool> registration_busy{false};
  std::queue<core::ImuControl> imu_buffer;
  std::queue<LidarScan> lidar_buffer;
  size_t max_lidar_buffer_size = 10;

  // metrics_dir set: each scan's pose (TUM) and statistics (CSV) are appended to two files every write_interval, so a
  // killed node loses at most that much. publish_metrics: the statistics are also published.
  std::string metrics_dir;
  std::chrono::seconds write_interval{60};
  std::mutex metrics_mutex;
  std::string pending_poses;
  std::string pending_scans;
  std::ofstream poses_file;
  std::ofstream scans_file;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr scan_metrics_publisher;
  // the threads last: the registration thread stops first, then the writer writes what is left
  std::jthread metrics_write_thread;
  std::jthread registration_thread;

  ThreadedNode() = delete;
  ThreadedNode(const std::string& node_name, const rclcpp::NodeOptions& options);

  void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr& imu_msg);
  void lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg);
  void registration_loop();

  // A scan's row: its statistics (empty for one dropped from the buffer) and what became of it; on ok, its pose.
  void record_scan(const LidarScan& scan,
                   std::chrono::steady_clock::time_point process_start,
                   ScanStatus status,
                   const core::ScanStats& stats);
  void write_metrics_loop(const std::stop_token& stop);

  ~ThreadedNode();
  ThreadedNode(const ThreadedNode&) = delete;
  ThreadedNode(ThreadedNode&&) = delete;
  ThreadedNode& operator=(const ThreadedNode&) = delete;
  ThreadedNode& operator=(ThreadedNode&&) = delete;
};

} // namespace rko_lio::ros

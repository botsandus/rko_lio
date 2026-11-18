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
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <vector>
// ros
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

namespace rko_lio::ros {

// What became of a scan: registered, skipped by the odometry (out of order, a fragment, or ending too soon after the
// previous one), dropped, or (merging) one revolution whose lidars' scans did not end together.
enum class ScanStatus : std::uint8_t { ok, skipped, buffer_full, input_error, registration_error, unsynced };

class ThreadedNode : public BaseNode {
public:
  std::mutex buffer_mutex;
  std::condition_variable sync_condition_variable;
  std::atomic<bool> atomic_can_process = false;
  std::atomic<bool> registration_busy{false};
  std::queue<core::ImuControl> imu_buffer;
  std::queue<LidarScan> lidar_buffer;
  size_t max_lidar_buffer_size = 10;

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr scan_metrics_publisher;

  // lidar_topics: each lidar's scans go into the base frame and are grouped by revolution (scan end relative to the
  // first one seen, in 10 Hz periods); a revolution's scans are concatenated with their own point times into one scan.
  // Only the lidar callbacks touch these; they share the node's default (mutually exclusive) callback group, so they
  // never run at once.
  struct MergeInput {
    std::string topic;
    std::optional<Sophus::SE3s> lidar2base; // from tf on its first scan: the lidars are rigid
    core::ScanFragmentCheck fragments;
  };
  std::vector<MergeInput> merge_inputs;
  std::map<std::int64_t, std::vector<std::optional<LidarScan>>> open_revolutions; // one slot per lidar
  std::optional<core::Nsec> revolution_origin;
  std::int64_t last_merged_revolution = std::numeric_limits<std::int64_t>::min();
  int consecutive_unsynced = 0;
  // stops before BaseNode's results writer, which then writes what is left
  std::jthread registration_thread;

  ThreadedNode() = delete;
  ThreadedNode(const std::string& node_name, const rclcpp::NodeOptions& options);

  void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr& imu_msg);
  void lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg);
  void merge_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg, std::size_t input);
  void merge_ready_revolutions();
  void merge_revolution(std::vector<std::optional<LidarScan>>& slots);
  void enqueue_scan(LidarScan scan);
  void registration_loop();

  // A scan's row: its statistics (empty for one dropped from the buffer) and what became of it.
  void record_scan(const LidarScan& scan,
                   std::chrono::steady_clock::time_point process_start,
                   ScanStatus status,
                   const core::ScanStats& stats);

  ~ThreadedNode();
  ThreadedNode(const ThreadedNode&) = delete;
  ThreadedNode(ThreadedNode&&) = delete;
  ThreadedNode& operator=(const ThreadedNode&) = delete;
  ThreadedNode& operator=(ThreadedNode&&) = delete;
};

} // namespace rko_lio::ros

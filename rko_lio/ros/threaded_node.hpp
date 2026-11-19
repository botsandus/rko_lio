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

// skipped: a fragment, a payload ending too soon after the previous, or a merge scan that arrived late or twice
enum class PayloadStatus : std::uint8_t { ok, skipped, buffer_full, input_error, registration_error, unsynced };

class ThreadedNode : public BaseNode {
public:
  std::mutex buffer_mutex;
  std::condition_variable sync_condition_variable;
  std::atomic<bool> atomic_can_process = false;
  std::atomic<bool> registration_busy{false};
  std::queue<core::ImuControl> imu_buffer;
  std::queue<LidarPayload> lidar_buffer;
  size_t max_lidar_buffer_size = 10;

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr payload_metrics_publisher;

  // merge state: touched only by the lidar callbacks, which never run at once (one mutually exclusive callback group)
  struct MultiLidarInput {
    std::string topic;
    std::optional<Sophus::SE3s> lidar2base; // from tf on its first scan: the lidars are rigid
    core::ScanFragmentCheck fragment_check;
  };
  struct PayloadSlot {
    std::optional<LidarPayload> scan;
    bool dropped = false; // its lidar's scan for this payload was a fragment
  };
  struct PayloadAnchor {
    core::Nsec end{0};
    std::int64_t index = 0;
  };
  std::vector<MultiLidarInput> multi_lidar_inputs;
  std::map<std::int64_t, std::vector<PayloadSlot>> open_payloads; // one slot per lidar
  std::optional<PayloadAnchor> payload_anchor;
  std::int64_t last_merged_payload_index = std::numeric_limits<std::int64_t>::min();
  int consecutive_unsynced = 0;
  std::jthread registration_thread;

  ThreadedNode() = delete;
  ThreadedNode(const std::string& node_name, const rclcpp::NodeOptions& options);

  void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr& imu_msg);
  void lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg);
  void multi_lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg, const std::size_t input);
  void merge_ready_payloads();
  void merge_payload(std::vector<PayloadSlot> slots);
  void add_to_lidar_buffer(LidarPayload payload);
  void registration_loop();

  // stats are empty for a payload dropped from the buffer
  void publish_payload_metrics(const LidarPayload& payload,
                               const std::chrono::steady_clock::time_point registration_start,
                               const PayloadStatus status,
                               const core::PayloadStats& stats);

  ~ThreadedNode();
  ThreadedNode(const ThreadedNode&) = delete;
  ThreadedNode(ThreadedNode&&) = delete;
  ThreadedNode& operator=(const ThreadedNode&) = delete;
  ThreadedNode& operator=(ThreadedNode&&) = delete;
};

} // namespace rko_lio::ros

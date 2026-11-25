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
#include "rko_lio/core/lio.hpp"
#include "rko_lio/ros/utils/point_cloud_read.hpp"
// stl
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>
// ros
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/u_int32.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <tf2_ros/transform_listener.hpp>

namespace rko_lio::ros {

// one scan per lidar: one, or several when merging
struct LidarPayload {
  std::vector<core::Timestamps> timestamps; // per cloud, each with its own min/max
  std::vector<core::Vector3sVector> clouds;
  std::vector<Sophus::SE3s> lidar2base; // per cloud
  // node clock minus the payload end, on entering rko_lio's callback, or merging, at the merge
  core::Nsec upstream{0};
  // merging: arrival and ready of the scan that completes the payload
  std::chrono::steady_clock::time_point arrival; // entering rko_lio's callback
  std::chrono::steady_clock::time_point ready;   // points converted from the message

  core::Nsec start() const { return core::payload_span(timestamps).start; }
  core::Nsec end() const { return core::payload_span(timestamps).end; }
  std::size_t point_count() const { return core::point_count(clouds); }
};

// published as a number in rko_lio/metrics
enum class PayloadStatus : std::uint8_t {
  ok,                 // registered, odometry published
  skipped,            // a fragment, a payload ending too soon after the previous, or a merge scan late or twice
  buffer_full,        // the oldest buffered payload, dropped at max_lidar_buffer_size
  input_error,        // core::InputError
  registration_error, // core::RegistrationError
  unsynced,           // merged scans whose ends differ by more than MAX_PHASE_SPREAD
};

class BaseNode {
public:
  // merge state: touched only by the lidar callbacks, which never run at once (one mutually exclusive callback group)
  struct MultiLidarInput {
    std::string topic;
    std::optional<Sophus::SE3s> lidar2base; // from tf on its first scan: the lidars are rigid
    core::ScanFragmentCheck fragment_check;
  };
  struct PayloadSlot {
    std::optional<LidarPayload> payload;
    bool dropped = false; // its lidar's scan for this payload was a fragment
  };
  struct PayloadAnchor {
    core::Nsec end{0};
    std::int64_t index = 0;
  };

  // from tf on the first message, its header's frame to the base frame
  std::optional<Sophus::SE3s> extrinsic_imu2base;
  std::optional<Sophus::SE3s> extrinsic_lidar2base;

  std::unique_ptr<core::LIO> lio;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
  core::Nsec publish_map_after = std::chrono::seconds(1);
  rclcpp::Node::SharedPtr node;
  utils::LidarTime lidar_time;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener; // after tf_buffer: its thread writes to it until destroyed

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr frame_publisher;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_publisher;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr reset_count_publisher;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr metrics_publisher;

  std::string imu_topic;
  std::string lidar_topic;
  std::vector<std::string> lidar_topics; // set instead of lidar_topic: these lidars are merged into one payload
  std::string base_frame;
  std::string odom_frame = "odom";
  std::string odom_topic = "rko_lio/odom";
  std::string map_topic = "rko_lio/local_map";
  std::string deskewed_scan_topic = "rko_lio/deskewed_scan";

  std::mutex local_map_mutex;

  bool publish_odom = true;
  bool publish_odom_tf = true;
  bool invert_odom_tf = false;
  bool publish_deskewed_scan = false;
  bool publish_local_map = false;
  bool reset_on_registration_error = false;

  // ~/reset sets it; the next registration resets the odometry first, on the registration thread
  std::atomic<bool> reset_requested = false;
  std::size_t reset_count = 0;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service;

  std::mutex buffer_mutex;
  std::condition_variable_any registration_wakeup;
  std::atomic<bool> can_register = false; // the front payload has an IMU sample past its end
  std::atomic<bool> registration_busy = false;
  std::queue<core::ImuControl> imu_buffer;
  std::queue<LidarPayload> lidar_buffer;
  std::size_t max_lidar_buffer_size = 10;

  std::vector<MultiLidarInput> multi_lidar_inputs;
  std::map<std::int64_t, std::vector<PayloadSlot>> open_payloads; // one slot per lidar
  std::optional<PayloadAnchor> payload_anchor;
  std::optional<std::int64_t> last_merged_payload_index;
  int consecutive_unsynced = 0;

  // last: joined (each stopped by its stop token) before the members they use are destroyed
  std::jthread map_publish_thread;
  std::jthread registration_thread;

  BaseNode(const std::string& node_name, const rclcpp::NodeOptions& options);
  BaseNode(const BaseNode&) = delete;
  BaseNode(BaseNode&&) = delete;
  BaseNode& operator=(const BaseNode&) = delete;
  BaseNode& operator=(BaseNode&&) = delete;
  ~BaseNode() = default;

  // nullopt until tf has the frame; throws std::invalid_argument for an empty frame
  std::optional<Sophus::SE3s> lookup_extrinsic(const std::string& frame) const;

  // stamps the arrival first; throws core::InputError
  LidarPayload lidar_msg_to_payload(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg,
                                    const Sophus::SE3s& lidar2base) const;

  void publish_odometry() const;
  void publish_tf() const;
  void publish_deskewed_clouds(const std::vector<core::Vector3sVector>& deskewed_clouds) const;
  void publish_map_loop(const std::stop_token& stop);

  void reset_odometry();

  void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr& imu_msg);
  void lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg);
  void multi_lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg, const std::size_t input);
  void merge_ready_payloads();
  void add_to_lidar_buffer(LidarPayload payload);
  void registration_loop(const std::stop_token& stop);

  void publish_metrics(const LidarPayload& payload,
                       const std::chrono::steady_clock::time_point registration_start,
                       const PayloadStatus status,
                       const core::PayloadStats& stats) const;
};

} // namespace rko_lio::ros

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
// stl
#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
// ros
#include <geometry_msgs/msg/accel_stamped.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/context.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
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

  core::Nsec start() const { return std::ranges::min(timestamps, {}, &core::Timestamps::min).min; }
  core::Nsec end() const { return std::ranges::max(timestamps, {}, &core::Timestamps::max).max; }
  std::size_t point_count() const {
    std::size_t count = 0;
    for (const core::Vector3sVector& cloud : clouds) {
      count += cloud.size();
    }
    return count;
  }
};
// Shared helper used by both the threaded and sequential nodes.
core::ImuControl imu_msg_to_imu_data(const sensor_msgs::msg::Imu& imu_msg);

// payload_metrics layout label, times in ns: upstream = LidarPayload::upstream; convert = arrival to ready; wait =
// ready to the registration start (an IMU sample past the payload end, or the previous payload); registration = start
// to the pose. end_ns as a double is rounded to 256 ns; the other values are exact
constexpr std::string_view PAYLOAD_METRICS_FIELDS = "end_ns,duration_ns,upstream_ns,convert_ns,wait_ns,"
                                                    "registration_ns,"
                                                    "status,input_points,keypoints,imu_samples,icp_iterations,"
                                                    "correspondences,icp_cost,reset_count";

class BaseNode {
public:
  // field order minimises padding
  // from tf on the first message, its header's frame to the base frame
  std::optional<Sophus::SE3s> extrinsic_imu2base;
  std::optional<Sophus::SE3s> extrinsic_lidar2base;

  std::unique_ptr<core::LIO> lio;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
  core::Nsec publish_map_after = std::chrono::seconds(1);
  rclcpp::Node::SharedPtr node;
  core::TimestampProcessingConfig timestamp_proc_config;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr frame_publisher;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_publisher;
  rclcpp::Publisher<geometry_msgs::msg::AccelStamped>::SharedPtr lidar_accel_publisher;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr reset_count_publisher;

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
  bool publish_lidar_acceleration = false;
  bool publish_deskewed_scan = false;
  bool publish_local_map = false;
  bool reset_on_registration_error = false;

  std::atomic<bool> atomic_node_running = true;
  // ~/reset sets it; the next registration resets the odometry first, on the registration thread
  std::atomic<bool> atomic_reset_requested = false;
  std::atomic<std::size_t> reset_count = 0;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service;
  std::jthread map_publish_thread;

  BaseNode() = delete;
  BaseNode(const std::string& node_name, const rclcpp::NodeOptions& options);

  void require_single_lidar() const; // throws std::invalid_argument if lidar_topics is set
  // false until tf has the frame; throws std::invalid_argument for an empty frame
  bool extrinsic_from_tf(std::optional<Sophus::SE3s>& extrinsic, const std::string& frame);

  // stamps the arrival first; throws core::InputError
  LidarPayload process_lidar_msg(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg,
                                 const Sophus::SE3s& lidar2base) const;

  std::vector<core::Vector3sVector> register_payload_locked(std::vector<core::Vector3sVector> clouds,
                                                            const std::span<const core::Timestamps> timestamps,
                                                            const std::span<const Sophus::SE3s> lidar2base);

  void publish_lidar_outputs(const std::vector<core::Vector3sVector>& deskewed_clouds) const;

  void publish_odometry(const core::State& state,
                        const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr& publisher) const;
  void publish_tf(const core::State& state) const;
  void publish_lidar_accel(const core::State& state) const;
  void publish_map_loop();

  void reset_odometry();

  ~BaseNode();
  BaseNode(const BaseNode&) = delete;
  BaseNode(BaseNode&&) = delete;
  BaseNode& operator=(const BaseNode&) = delete;
  BaseNode& operator=(BaseNode&&) = delete;
};

} // namespace rko_lio::ros

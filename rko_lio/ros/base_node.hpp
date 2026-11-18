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
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
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

// One scan as one cloud per lidar (one, or one per merged lidar), never concatenated.
struct LidarScan {
  std::vector<core::Timestamps> timestamps; // per cloud; every entry carries the whole scan's min/max
  std::vector<core::Vector3sVector> points;
  // per cloud: from its lidar's frame, which its points are in, to the base frame
  std::vector<Sophus::SE3s> lidar2base;
  core::Nsec transport{0};                       // node clock at arrival minus the scan's last point
  std::chrono::steady_clock::time_point arrival; // when the callback took the message

  core::Nsec start() const { return timestamps.front().min; }
  core::Nsec end() const { return timestamps.front().max; }
  std::size_t point_count() const {
    std::size_t count = 0;
    for (const core::Vector3sVector& cloud : points) {
      count += cloud.size();
    }
    return count;
  }
};
// Shared helper used by both the threaded and sequential nodes.
core::ImuControl imu_msg_to_imu_data(const sensor_msgs::msg::Imu& imu_msg);

// A scan's statistics, as the published message's layout label. Times in ns: the scan's last point (sensor clock) and
// duration; transport: its arrival on the node clock minus its last point; queue and process: the steady clock from
// arrival to the start of its processing, and the processing.
constexpr std::string_view SCAN_STATISTICS_FIELDS = "scan_end_ns,scan_duration_ns,transport_ns,queue_ns,process_ns,"
                                                    "status,input_points,keypoints,imu_samples,icp_iterations,"
                                                    "correspondences,icp_cost,reset_count";

class BaseNode {
public:
  // field order minimises padding
  Sophus::SE3s extrinsic_imu2base;
  Sophus::SE3s extrinsic_lidar2base;

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
  std::string imu_frame; // default: get from the first imu message
  std::string lidar_topic;
  std::vector<std::string> lidar_topics; // set instead of lidar_topic: these lidars are merged into one scan
  std::string lidar_frame;               // default: get from the first lidar message
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
  bool extrinsics_set = false;
  bool reset_on_registration_error = false;

  std::atomic<bool> atomic_node_running = true;
  // ~/reset sets it; the next registration resets the odometry first, on the registration thread
  std::atomic<bool> atomic_reset_requested = false;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service;
  std::jthread map_publish_thread;

  BaseNode() = delete;
  BaseNode(const std::string& node_name, const rclcpp::NodeOptions& options);

  void parse_cli_extrinsics();
  void require_single_lidar() const; // throws std::invalid_argument if lidar_topics is set
  bool check_and_set_extrinsics();

  // Parse the message's frame_id into target_frame on first sight, throw if neither header nor static extrinsics
  // are usable, and report whether extrinsics are now set. Extrinsics are assumed static; if they change, switch
  // to querying TF inline per-message.
  bool ensure_frame_and_extrinsics(std::string& target_frame, const std::string& msg_frame, std::string_view kind);

  LidarScan process_lidar_msg(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg) const;

  std::vector<core::Vector3sVector> register_scan_locked(std::vector<core::Vector3sVector> scan,
                                                         const std::span<const core::Timestamps> timestamps,
                                                         const std::span<const Sophus::SE3s> lidar2base);

  void publish_lidar_outputs(const std::vector<core::Vector3sVector>& deskewed_scan) const;

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

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
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <span>
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
  skipped,            // a fragment, a payload ending too soon after the previous, or a lidar scan the merge drops
  buffer_full,        // the oldest buffered payload, dropped at max_lidar_buffer_size
  input_error,        // core::InputError
  registration_error, // core::RegistrationError
  unsynced,           // a merged payload whose scans are stamped more than MAX_STAMP_SPREAD apart
};

// One per IMU topic (imu_topic, or each entry of imu_topics). imu2base is written once, by that IMU's callback, before
// its first sample is buffered. All other fields are guarded by buffer_mutex.
struct ImuInput {
  std::string topic;
  std::optional<Sophus::SE3s> imu2base; // looked up from tf once, on the first sample
  std::deque<core::ImuControl> buffer;
  std::optional<core::Nsec> newest_time;
  bool stale = false; // no sample yet, or newest_time more than MAX_IMU_LAG older than the newest IMU sample
  std::optional<core::Nsec> stale_from; // newest_time when it went stale
  // Reset every LATE_IMU_LOG_PERIOD. used_samples were fed to the LIO, late_samples were dropped as older than
  // fed_until.
  std::size_t used_samples = 0;
  std::size_t late_samples = 0;
};

// Whether a payload ending at end can register. Some IMU must have a sample past end. In addition, either every IMU
// that is not stale has a sample past end, or min_samples is set and the buffers hold at least that many samples
// stamped in [fed_until, end). inputs must not be empty.
bool imus_cover(const std::span<const ImuInput> inputs,
                const core::Nsec fed_until,
                const core::Nsec end,
                const std::optional<std::size_t> min_samples);

class BaseNode {
public:
  // merge state: touched only by the lidar callbacks, which never run at once (one mutually exclusive callback group)
  struct PayloadAnchor {
    core::Nsec stamp{0};    // header stamp of a scan
    std::int64_t index = 0; // payload index given to the scan, as in the keys of open_payloads
  };
  struct MultiLidarInput {
    std::string topic;
    std::optional<Sophus::SE3s> lidar2base; // from tf on its first scan: the lidars are rigid
    core::ScanFragmentCheck fragment_check;
    // the latest scan of this lidar given a payload index
    std::optional<PayloadAnchor> anchor;
  };
  struct PayloadSlot {
    std::optional<LidarPayload> payload;
    core::Nsec stamp{0}; // header stamp of the slot's scan
  };

  // from tf on the first message, its header's frame to the base frame
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
  // Set online: a payload can register on this many samples before every IMU is past its end. Unset offline: a payload
  // waits for every IMU that is not stale, so a run on one bag gives the same odometry every time.
  const std::optional<std::size_t> min_imu_samples;
  std::atomic<bool> can_register = false; // imus_cover() holds for the oldest buffered payload
  std::atomic<bool> registration_busy = false;
  std::vector<ImuInput> imu_inputs;
  // End of the last payload whose IMU samples were fed to the LIO. A sample stamped before it is late, and is dropped
  // and counted.
  core::Nsec fed_until{0};
  std::queue<LidarPayload> lidar_buffer;
  std::size_t max_lidar_buffer_size = 10;

  std::vector<MultiLidarInput> multi_lidar_inputs;
  std::map<std::int64_t, std::vector<PayloadSlot>> open_payloads; // one slot per lidar
  // the first scan given the highest payload index so far
  std::optional<PayloadAnchor> payload_anchor;
  std::optional<std::int64_t> last_merged_payload_index;
  std::chrono::steady_clock::time_point late_imu_samples_logged; // registration thread only

  // last: joined (each stopped by its stop token) before the members they use are destroyed
  std::jthread map_publish_thread;
  std::jthread registration_thread;

  BaseNode(const std::string& node_name, const rclcpp::NodeOptions& options, const bool wait_for_every_imu);
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

  void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr& imu_msg, const std::size_t input);
  // Call with buffer_mutex held. Updates each IMU's stale flag, logs when an IMU becomes stale or recovers, and sets
  // can_register.
  void update_can_register();
  // Call with buffer_mutex held. Feeds the LIO every buffered sample stamped before the payload end, in stamp order
  // across IMUs. Samples stamped before fed_until are dropped and counted as late.
  void feed_imu_before(const core::Nsec end);
  void lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg);
  // Merges the lidars of lidar_topics, which share one clock and each scan once per 100 ms at any phase. Scans are
  // paired by header stamp into payloads of at most one scan per lidar. A scan or payload that does not fit is dropped.
  void multi_lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg, const std::size_t input);
  void merge_ready_payloads();
  void restart_merge();
  void add_to_lidar_buffer(LidarPayload payload);
  void registration_loop(const std::stop_token& stop);

  void publish_metrics(const LidarPayload& payload,
                       const std::chrono::steady_clock::time_point registration_start,
                       const PayloadStatus status,
                       const core::PayloadStats& stats) const;
};

} // namespace rko_lio::ros

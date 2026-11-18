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

#include "base_node.hpp"
#include "rko_lio/ros/utils/spdlog_sink.hpp"
#include "rko_lio/ros/utils/utils.hpp"
#include <UTL/profiler.hpp>
// other
#include <optional>
#include <rclcpp/version.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace {
using namespace std::literals;
} // namespace

namespace rko_lio::ros {

core::ImuControl imu_msg_to_imu_data(const sensor_msgs::msg::Imu& imu_msg) {
  core::ImuControl imu_data;
  imu_data.time = utils::to_ns(imu_msg.header.stamp);
  imu_data.angular_velocity = utils::ros_xyz_to_eigen_vector(imu_msg.angular_velocity);
  imu_data.acceleration = utils::ros_xyz_to_eigen_vector(imu_msg.linear_acceleration);
  return imu_data;
}

BaseNode::BaseNode(const std::string& node_name, const rclcpp::NodeOptions& options) {
  node = rclcpp::Node::make_shared(node_name, options);
  auto logger = std::make_shared<spdlog::logger>("rko_lio", std::make_shared<utils::RclcppSink>(node->get_logger()));
  logger->set_level(spdlog::level::trace);
  spdlog::set_default_logger(std::move(logger));

  imu_topic = node->declare_parameter<std::string>("imu_topic"); // required
  lidar_topic = node->declare_parameter<std::string>("lidar_topic", lidar_topic);
  lidar_topics = node->declare_parameter<std::vector<std::string>>("lidar_topics", lidar_topics);
  if (lidar_topic.empty() == lidar_topics.empty()) {
    throw std::invalid_argument(
        "Set exactly one of lidar_topic and lidar_topics (several lidars merged into one scan).");
  }
  base_frame = node->declare_parameter<std::string>("base_frame"); // required
  imu_frame = node->declare_parameter<std::string>("imu_frame", imu_frame);
  lidar_frame = node->declare_parameter<std::string>("lidar_frame", lidar_frame);
  if (!lidar_topics.empty()) {
    if (!lidar_frame.empty()) {
      throw std::invalid_argument(
          "lidar_frame is per lidar: with lidar_topics each lidar's frame comes from its scans.");
    }
    lidar_frame = base_frame; // each merged cloud carries its own lidar's extrinsic
  }
  odom_frame = node->declare_parameter<std::string>("odom_frame", odom_frame);
  odom_topic = node->declare_parameter<std::string>("odom_topic", odom_topic);

  // tf
  invert_odom_tf = node->declare_parameter<bool>("invert_odom_tf", invert_odom_tf);
  tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
#if RCLCPP_VERSION_MAJOR >= 30
  tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer, *node);
#else
  tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer, node);
#endif

  // publishing
  const rclcpp::QoS publisher_qos(rclcpp::SystemDefaultsQoS().keep_last(1).durability_volatile());
  publish_odom = node->declare_parameter<bool>("publish_odom", publish_odom);
  publish_odom_tf = node->declare_parameter<bool>("publish_odom_tf", publish_odom_tf);
  if (publish_odom_tf) {
    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*node);
  }
  if (publish_odom) {
    odom_publisher = node->create_publisher<nav_msgs::msg::Odometry>(odom_topic, publisher_qos);
    reset_count_publisher = node->create_publisher<std_msgs::msg::UInt32>(
        "rko_lio/reset_count", rclcpp::SystemDefaultsQoS().keep_last(1).transient_local());
    reset_count_publisher->publish(std_msgs::msg::UInt32());
  }

  publish_lidar_acceleration = node->declare_parameter<bool>("publish_lidar_acceleration", publish_lidar_acceleration);
  if (publish_lidar_acceleration) {
    lidar_accel_publisher =
        node->create_publisher<geometry_msgs::msg::AccelStamped>("rko_lio/lidar_acceleration", publisher_qos);
  }

  publish_deskewed_scan = node->declare_parameter<bool>("publish_deskewed_scan", publish_deskewed_scan);
  if (publish_deskewed_scan) {
    deskewed_scan_topic = node->declare_parameter<std::string>("deskewed_scan_topic", deskewed_scan_topic);
    frame_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(deskewed_scan_topic, publisher_qos);
  }

  publish_local_map = node->declare_parameter<bool>("publish_local_map", publish_local_map);
  if (publish_local_map) {
    map_topic = node->declare_parameter<std::string>("map_topic", map_topic);
    const double publish_map_after_seconds =
        node->declare_parameter<double>("publish_map_after", core::to_seconds(publish_map_after));
    publish_map_after =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(publish_map_after_seconds));
    map_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(map_topic, publisher_qos);
  }

  // lio params
  core::LIO::Config lio_config{};
  lio_config.deskew = node->declare_parameter<bool>("deskew", lio_config.deskew);
  lio_config.max_iterations = static_cast<size_t>(node->declare_parameter<int>("max_iterations", 50));
  lio_config.voxel_size =
      static_cast<core::Scalar>(node->declare_parameter<double>("voxel_size", lio_config.voxel_size));
  if (node->declare_parameter<int>("max_points_per_voxel", -1) != -1) {
    throw std::invalid_argument(
        "max_points_per_voxel was removed, it is now fixed at compile time so the map can store points inline. "
        "Use voxel_size to control map density instead: smaller keeps more points overall, larger keeps fewer.");
  }
  lio_config.max_range = static_cast<core::Scalar>(node->declare_parameter<double>("max_range", lio_config.max_range));
  lio_config.min_range = static_cast<core::Scalar>(node->declare_parameter<double>("min_range", lio_config.min_range));
  lio_config.convergence_criterion = static_cast<core::Scalar>(
      node->declare_parameter<double>("convergence_criterion", lio_config.convergence_criterion));
  lio_config.max_correspondence_distance = static_cast<core::Scalar>(
      node->declare_parameter<double>("max_correspondence_distance", lio_config.max_correspondence_distance));
  lio_config.max_num_threads = static_cast<int>(node->declare_parameter<int>("max_num_threads", 1));
  lio_config.initialization_phase =
      node->declare_parameter<bool>("initialization_phase", lio_config.initialization_phase);
  lio_config.double_downsample = node->declare_parameter<bool>("double_downsample", lio_config.double_downsample);
  lio_config.flat = node->declare_parameter<bool>("flat", lio_config.flat);
  lio = std::make_unique<core::LIO>(lio_config);

  // Lidar per-point timestamp processing params, namespaced under lidar_timestamps.*
  timestamp_proc_config.multiplier_to_seconds = node->declare_parameter<double>(
      "lidar_timestamps.multiplier_to_seconds", timestamp_proc_config.multiplier_to_seconds);
  timestamp_proc_config.force_absolute =
      node->declare_parameter<bool>("lidar_timestamps.force_absolute", timestamp_proc_config.force_absolute);
  timestamp_proc_config.force_relative =
      node->declare_parameter<bool>("lidar_timestamps.force_relative", timestamp_proc_config.force_relative);

  // manually, if, define extrinsics
  parse_cli_extrinsics();

  RCLCPP_INFO_STREAM(node->get_logger(),
                     "Subscribed to IMU: "
                         << imu_topic << (!imu_frame.empty() ? " (frame " + imu_frame + ")" : "") << " and LiDAR: "
                         << (lidar_topics.empty() ? lidar_topic
                                                  : "a merge of " + std::to_string(lidar_topics.size()) + " lidars")
                         << (!lidar_frame.empty() ? " (frame " + lidar_frame + ")" : "")
                         << ". Max number of threads: " << lio_config.max_num_threads << ". "
                         << (publish_odom ? "Publishing odometry to " + odom_topic + " ( " + odom_frame + " )"
                                          : std::string("Not publishing odometry"))
                         << (publish_odom_tf ? ", its tf" : ", no tf")
                         << " and acceleration "
                            "estimates to rko_lio/lidar_acceleration. Deskewing is "
                         << (lio->config.deskew ? "enabled" : "disabled") << "."
                         << (publish_deskewed_scan ? (" Publishing the deskewed scan to " + deskewed_scan_topic + ".")
                                                   : ""));

  reset_on_registration_error =
      node->declare_parameter<bool>("reset_on_registration_error", reset_on_registration_error);
  reset_service = node->create_service<std_srvs::srv::Trigger>(
      "~/reset", [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>&,
                        const std::shared_ptr<std_srvs::srv::Trigger::Response>& response) {
        atomic_reset_requested = true;
        response->success = true;
        response->message = "The odometry resets before its next scan.";
      });

  if (publish_local_map) { // last: a throw above must not leave this thread running, and it reads lio
    map_publish_thread = std::jthread([this] { publish_map_loop(); });
  }
  RCLCPP_INFO(node->get_logger(), "RKO LIO Node is up!");
}

void BaseNode::parse_cli_extrinsics() {
  const auto parse_extrinsic = [this](const std::string& name, Sophus::SE3s& extrinsic) {
    const std::string param_name = "extrinsic_" + name + "2base_quat_xyzw_xyz";
    const std::vector<double> vec = node->declare_parameter<std::vector<double>>(param_name, std::vector<double>{});
    if (vec.empty()) {
      return false;
    }
    try {
      extrinsic = utils::to_se3(vec);
    } catch (const core::InputError& error) {
      throw core::InputError("Parameter '" + param_name + "' is invalid. " + error.what());
    }
    RCLCPP_INFO_STREAM(node->get_logger(), "Parsed " << name << " extrinsic as: " << extrinsic.log().transpose());
    return true;
  };
  const bool imu_ok = parse_extrinsic("imu", extrinsic_imu2base);
  const bool lidar_ok = parse_extrinsic("lidar", extrinsic_lidar2base);
  if (!lidar_topics.empty()) {
    if (lidar_ok) {
      throw std::invalid_argument(
          "extrinsic_lidar2base_quat_xyzw_xyz is per lidar: with lidar_topics each lidar's extrinsic comes from tf.");
    }
    extrinsics_set = imu_ok; // merging: each cloud's extrinsic comes from tf
    return;
  }
  extrinsics_set = imu_ok && lidar_ok;
}

void BaseNode::require_single_lidar() const {
  if (!lidar_topics.empty()) {
    throw std::invalid_argument(
        "lidar_topics (several lidars merged into one scan) is supported by the online node only.");
  }
}

bool BaseNode::ensure_frame_and_extrinsics(std::string& target_frame,
                                           const std::string& msg_frame,
                                           std::string_view kind) {
  if (target_frame.empty()) {
    if (msg_frame.empty() && !extrinsics_set) {
      throw std::invalid_argument(std::string(kind) +
                                  " message header has no frame id and we need it to query TF for the extrinsics. "
                                  "Either specify the frame id or the extrinsic manually.");
    }
    target_frame = msg_frame;
    RCLCPP_INFO_STREAM(node->get_logger(), "Parsed the " << kind << " frame id as: " << target_frame);
  }
  return check_and_set_extrinsics();
}

bool BaseNode::check_and_set_extrinsics() {
  if (extrinsics_set) {
    return true;
  }
  const std::optional<Sophus::SE3s> imu_transform = utils::get_transform(tf_buffer, imu_frame, base_frame, 0s);
  if (!imu_transform) {
    return false;
  }
  const std::optional<Sophus::SE3s> lidar_transform = utils::get_transform(tf_buffer, lidar_frame, base_frame, 0s);
  if (!lidar_transform) {
    return false;
  }
  extrinsic_imu2base = imu_transform.value();
  extrinsic_lidar2base = lidar_transform.value();
  extrinsics_set = true;
  return true;
}

LidarScan BaseNode::process_lidar_msg(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg) const {
  const auto arrival = std::chrono::steady_clock::now();
  const core::Nsec arrival_node_clock = utils::to_ns(node->now());
  UTL_PROFILER_SCOPE("rko_lio: convert message");
  const core::Nsec header_stamp = utils::to_ns(lidar_msg->header.stamp);
  LidarScan scan{.arrival = arrival};
  if (lio->config.deskew) {
    utils::RawScan raw = utils::point_cloud2_to_eigen_with_timestamps(lidar_msg);
    scan.timestamps.push_back(core::process_timestamps(raw.timestamps, header_stamp, timestamp_proc_config));
    scan.points.push_back(std::move(raw.points));
  } else {
    RCLCPP_WARN_STREAM_ONCE(node->get_logger(),
                            "Deskewing is disabled. Populating timestamps with static header time.");
    scan.points.push_back(utils::point_cloud2_to_eigen(lidar_msg));
    scan.timestamps.push_back({.min = header_stamp, .max = header_stamp});
    scan.timestamps.front().per_point.assign(scan.points.front().size(), header_stamp);
  }
  scan.lidar2base.push_back(extrinsic_lidar2base);
  scan.upstream = arrival_node_clock - scan.end();
  return scan;
}

std::vector<core::Vector3sVector> BaseNode::register_scan_locked(std::vector<core::Vector3sVector> scan,
                                                                 const std::span<const core::Timestamps> timestamps,
                                                                 const std::span<const Sophus::SE3s> lidar2base) {
  if (atomic_reset_requested.exchange(false)) {
    RCLCPP_WARN(node->get_logger(), "Odometry reset requested through the reset service.");
    reset_odometry(); // takes local_map_mutex itself
  }
  std::unique_lock lock(local_map_mutex, std::defer_lock);
  if (publish_local_map) {
    lock.lock(); // map publish thread may access map simultaneously
  }
  return lio->register_scan(std::move(scan), timestamps, lidar2base);
}

void BaseNode::publish_lidar_outputs(const std::vector<core::Vector3sVector>& deskewed_scan) const {
  publish_odometry(lio->lidar_state, odom_publisher);
  if (publish_lidar_acceleration) {
    publish_lidar_accel(lio->lidar_state);
  }
  if (publish_deskewed_scan) {
    std_msgs::msg::Header header;
    header.frame_id = base_frame;
    header.stamp = utils::to_ros_time(lio->lidar_state.time);
    core::Vector3sVector points; // one message for all clouds
    for (const core::Vector3sVector& cloud : deskewed_scan) {
      points.insert(points.end(), cloud.begin(), cloud.end());
    }
    frame_publisher->publish(utils::eigen_to_point_cloud2(points, header));
  }
}

void BaseNode::publish_odometry(const core::State& state,
                                const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr& publisher) const {
  if (!publish_odom) {
    return;
  }
  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header.stamp = utils::to_ros_time(state.time);
  odom_msg.header.frame_id = odom_frame;
  odom_msg.child_frame_id = base_frame;
  odom_msg.pose.pose = utils::sophus_to_pose(state.pose);
  utils::eigen_vector_to_ros_xyz(state.linear_velocity, odom_msg.twist.twist.linear);
  utils::eigen_vector_to_ros_xyz(state.angular_velocity, odom_msg.twist.twist.angular);
  publisher->publish(odom_msg);
}

void BaseNode::publish_tf(const core::State& state) const {
  if (!publish_odom_tf) {
    return;
  }
  geometry_msgs::msg::TransformStamped transform_msg;
  transform_msg.header.stamp = utils::to_ros_time(state.time);
  if (invert_odom_tf) {
    transform_msg.header.frame_id = base_frame;
    transform_msg.child_frame_id = odom_frame;
    transform_msg.transform = utils::sophus_to_transform(state.pose.inverse());
  } else {
    transform_msg.header.frame_id = odom_frame;
    transform_msg.child_frame_id = base_frame;
    transform_msg.transform = utils::sophus_to_transform(state.pose);
  }
  tf_broadcaster->sendTransform(transform_msg);
}

void BaseNode::publish_lidar_accel(const core::State& state) const {
  auto accel_msg = geometry_msgs::msg::AccelStamped();
  accel_msg.header.stamp = utils::to_ros_time(state.time);
  accel_msg.header.frame_id = base_frame;
  utils::eigen_vector_to_ros_xyz(state.linear_acceleration, accel_msg.accel.linear);
  lidar_accel_publisher->publish(accel_msg);
}

void BaseNode::publish_map_loop() {
  while (atomic_node_running) {
    std::this_thread::sleep_for(publish_map_after);
    std::unique_lock lock(local_map_mutex);
    if (lio->map.empty()) {
      RCLCPP_WARN_ONCE(node->get_logger(), "Local map publish thread: Local map is empty.");
      continue;
    }
    const core::Vector3sVector map_points = lio->map.points();
    const core::Nsec map_time = lio->lidar_state.time;
    lock.unlock(); // we don't access the local map anymore
    std_msgs::msg::Header map_header;
    map_header.stamp = utils::to_ros_time(map_time);
    map_header.frame_id = odom_frame;
    map_publisher->publish(utils::eigen_to_point_cloud2(map_points, map_header));
  }
}

BaseNode::~BaseNode() { atomic_node_running = false; }

void BaseNode::reset_odometry() {
  const bool forced_init_off = lio->config.initialization_phase;
  std::unique_lock lock(local_map_mutex, std::defer_lock);
  if (publish_local_map) {
    lock.lock();
  }
  lio->restart();
  reset_count = lio->reset_count;
  RCLCPP_WARN_STREAM(node->get_logger(), "Odometry reset count: " << reset_count);
  if (publish_odom) {
    std_msgs::msg::UInt32 reset_count_msg;
    reset_count_msg.data = static_cast<std::uint32_t>(reset_count);
    reset_count_publisher->publish(reset_count_msg);
  }
  if (forced_init_off) {
    RCLCPP_WARN(node->get_logger(),
                "initialization_phase was enabled; it is off after reset because static start is not guaranteed.");
  }
}

} // namespace rko_lio::ros

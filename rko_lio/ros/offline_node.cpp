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
#include "rko_lio/ros/utils/point_cloud_read.hpp"
#include "rko_lio/ros/utils/rosbag.hpp"
#include <UTL/profiler.hpp>
#include <algorithm>
#include <cstddef>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <vector>
// other
#include <std_msgs/msg/float32_multi_array.hpp>

namespace rko_lio::ros {
namespace {
using BagProgressPublisher = rclcpp::Publisher<std_msgs::msg::Float32MultiArray>;

class OfflineNode : public BaseNode {
public:
  const std::unique_ptr<utils::BufferableBag> bag;

  BagProgressPublisher::SharedPtr bag_progress_publisher;

  const float total_bag_msgs;
  float processed_bag_msgs = 0;
  const std::chrono::steady_clock::time_point bag_start_time;

  // called while the bag member is being constructed, so it may only read BaseNode members
  std::vector<std::string> bag_topics() const {
    std::vector<std::string> topics;
    for (const ImuInput& in : imu_inputs) {
      topics.push_back(in.topic);
    }
    topics.push_back(lidar_topic);
    return topics;
  }

  explicit OfflineNode(const rclcpp::NodeOptions& options)
      : BaseNode("rko_lio_offline_node", options, /*wait_for_every_imu=*/true),
        bag(std::make_unique<utils::BufferableBag>(
            node->declare_parameter<std::string>("bag_path"),
            bag_topics(),
            tf_buffer,
            tf2::durationFromSec(node->declare_parameter<double>("skip_first_seconds", 0.0)))),
        total_bag_msgs(static_cast<float>(bag->message_count())),
        bag_start_time(std::chrono::steady_clock::now()) {
    if (!lidar_topics.empty()) {
      throw std::invalid_argument(
          "lidar_topics (several lidars merged into one payload) is supported by the online node only.");
    }
    bag_progress_publisher = node->create_publisher<std_msgs::msg::Float32MultiArray>("rko_lio/bag_progress", 10);
  }

  void publish_bag_progress() const {
    const auto now = std::chrono::steady_clock::now();
    const float elapsed_seconds = std::chrono::duration<float>(now - bag_start_time).count();

    const float percent_complete = 100.0F * processed_bag_msgs / total_bag_msgs;
    const float avg_time_per_msg = (processed_bag_msgs > 0) ? elapsed_seconds / processed_bag_msgs : 0.0F;
    const float seconds_remaining = avg_time_per_msg * (total_bag_msgs - processed_bag_msgs);

    std_msgs::msg::Float32MultiArray progress_msg;
    progress_msg.layout.dim.resize(1);
    progress_msg.layout.dim.at(0).label = "percent_complete,seconds_remaining";
    progress_msg.layout.dim.at(0).size = 2;
    progress_msg.layout.dim.at(0).stride = 2;
    progress_msg.data = {percent_complete, seconds_remaining};

    bag_progress_publisher->publish(progress_msg);
  }

  void run() {
    const utils::LidarDeserializer deserialize_lidar(bag->topic_type(lidar_topic));
    while (rclcpp::ok() && !bag->finished()) {
      {
        std::size_t buffered_frames = 0;
        bool registration_can_drain = false;
        {
          const std::scoped_lock lock(buffer_mutex);
          buffered_frames = lidar_buffer.size();
          registration_can_drain = can_register.load() || registration_busy.load();
        }
        if (2 * buffered_frames >= max_lidar_buffer_size && registration_can_drain) {
          RCLCPP_WARN_STREAM_ONCE(node->get_logger(),
                                  "Lidar buffer size: " << buffered_frames
                                                        << ", max_lidar_buffer_size: " << max_lidar_buffer_size
                                                        << ", throttling the bag reading thread as it's too fast.\n");
          // this is a hack. can be improved
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
          continue;
        }
      }
      const rosbag2_storage::SerializedBagMessage serialized_bag_msg = bag->PopNextMessage();
      const auto& topic_name = serialized_bag_msg.topic_name;

      // check the topic and call the appropriate callback
      const auto imu = std::ranges::find(imu_inputs, topic_name, &ImuInput::topic);
      if (imu != imu_inputs.end()) {
        imu_callback(
            utils::deserialize<sensor_msgs::msg::Imu>(rclcpp::SerializedMessage(*serialized_bag_msg.serialized_data)),
            static_cast<std::size_t>(imu - imu_inputs.begin()));
      } else if (topic_name == lidar_topic) {
        if (const auto lidar_msg =
                deserialize_lidar(std::make_shared<rclcpp::SerializedMessage>(*serialized_bag_msg.serialized_data))) {
          lidar_callback(lidar_msg);
        }
      }

      processed_bag_msgs++;
      publish_bag_progress();
    }
    while (rclcpp::ok()) {
      {
        // wait for the registration buffer to drain - leftover IMU after the last lidar scan is harmless
        const std::scoped_lock lock(buffer_mutex);
        if (!can_register.load() && !registration_busy.load()) {
          break;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
};
} // namespace
} // namespace rko_lio::ros

int main(int argc, char* const* argv) {
  try {
    rclcpp::init(argc, argv);
    auto node = rko_lio::ros::OfflineNode(rclcpp::NodeOptions());
    {
      UTL_PROFILER_SCOPE("rko_lio: offline run");
      node.run();
    }
    rclcpp::shutdown();
  } catch (const std::exception& error) {
    spdlog::critical("{}", error.what());
    return 1;
  }
  return 0;
}

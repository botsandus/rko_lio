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

#include <rko_lio/core/util.hpp>

#include <geometry_msgs/msg/vector3.hpp>

namespace rko_lio::ros::utils {
inline Eigen::Vector3s ros_to_eigen_vector3(const geometry_msgs::msg::Vector3& vector) {
  return Eigen::Vector3d(vector.x, vector.y, vector.z).cast<core::Scalar>();
}

inline geometry_msgs::msg::Vector3 eigen_to_ros_vector3(const Eigen::Vector3s& vector) {
  geometry_msgs::msg::Vector3 ros_vector;
  ros_vector.x = vector.x();
  ros_vector.y = vector.y();
  ros_vector.z = vector.z();
  return ros_vector;
}
} // namespace rko_lio::ros::utils

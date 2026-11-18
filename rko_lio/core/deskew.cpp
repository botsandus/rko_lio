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

#include "deskew.hpp"

#include <algorithm>
#include <cmath>
#include <sophus/se3.hpp>
#include <tsl/robin_map.h>

namespace {
using namespace rko_lio::core;

// SE(3) exp about a fixed axis over dt, as R * p + V * rho
// Rodrigues -> K = hat(axis), R = I + sin(theta) K + (1 - cos theta) K^2
// left Jacobian -> V = I + ((1 - cos theta) / theta) K + ((theta - sin theta) / theta) K^2
// rho = dt * mean_linear_velocity and theta = angular_speed * dt
// dt cancels in V * rho
struct AxisExp {
  Eigen::Matrix3s R;
  Eigen::Vector3s V_times_rho;
};
} // namespace

namespace rko_lio::core {
void deskew_scan(Vector3sVector& scan,
                 const Timestamps& timestamps,
                 const Nsec reference_time,
                 const MotionPrior& motion,
                 const Sophus::SE3s& lidar2base,
                 const bool per_stamp) {
  const Sophus::SE3s start2reference = relative_pose_at_time(motion, reference_time).inverse();
  const Eigen::Matrix3s start2reference_rotation = start2reference.so3().matrix();
  const Eigen::Vector3s& start2reference_translation = start2reference.translation();
  const Eigen::Matrix3s lidar2base_rotation = lidar2base.so3().matrix();
  const Eigen::Vector3s& lidar2base_translation = lidar2base.translation();

  // to avoid numerical issues
  const Scalar angular_speed = std::max(motion.angular_velocity.norm(), static_cast<Scalar>(1e-6));
  const Scalar inverse_angular_speed = 1 / angular_speed;
  const Eigen::Vector3s axis = motion.angular_velocity * inverse_angular_speed;
  const Eigen::Matrix3s K = Sophus::SO3s::hat(axis);
  const Eigen::Matrix3s K_squared = K * K;

  const auto exp_at = [&](const Nsec timestamp) {
    const auto dt = to_seconds<Scalar>(timestamp - motion.start_time);
    const Scalar theta = angular_speed * dt;
    const Scalar sin_theta = std::sin(theta);
    const Scalar one_minus_cos_theta = 1 - std::cos(theta);
    const Eigen::Vector3s mean_linear_velocity = motion.linear_velocity + motion.acceleration * dt / 2;
    const Eigen::Vector3s K_times_linear_velocity = axis.cross(mean_linear_velocity);
    return AxisExp{
        .R = Eigen::Matrix3s::Identity() + sin_theta * K + one_minus_cos_theta * K_squared,
        .V_times_rho = dt * mean_linear_velocity +
                       (one_minus_cos_theta * inverse_angular_speed) * K_times_linear_velocity +
                       ((theta - sin_theta) * inverse_angular_speed) * axis.cross(K_times_linear_velocity),
    };
  };

  if (!per_stamp) {
    std::transform(scan.cbegin(), scan.cend(), timestamps.per_point.cbegin(), scan.begin(),
                   [&](const Eigen::Vector3s& lidar_point, const Nsec timestamp) {
                     const AxisExp exp = exp_at(timestamp);
                     const Eigen::Vector3s point = lidar2base_rotation * lidar_point + lidar2base_translation;
                     return Eigen::Vector3s(start2reference_rotation * (exp.R * point + exp.V_times_rho) +
                                            start2reference_translation);
                   });
    return;
  }

  tsl::robin_map<Nsec::rep, Eigen::Matrix<Scalar, 3, 4>> lidar2reference_by_stamp;
  lidar2reference_by_stamp.reserve(1024);
  std::transform(scan.cbegin(), scan.cend(), timestamps.per_point.cbegin(), scan.begin(),
                 [&](const Eigen::Vector3s& lidar_point, const Nsec timestamp) {
                   auto [entry, inserted] = lidar2reference_by_stamp.try_emplace(timestamp.count());
                   if (inserted) {
                     const AxisExp exp = exp_at(timestamp);
                     Eigen::Matrix<Scalar, 3, 4>& lidar2reference = entry.value();
                     lidar2reference.leftCols<3>() = start2reference_rotation * exp.R * lidar2base_rotation;
                     lidar2reference.col(3) =
                         start2reference_rotation * (exp.R * lidar2base_translation + exp.V_times_rho) +
                         start2reference_translation;
                   }
                   const Eigen::Matrix<Scalar, 3, 4>& lidar2reference = entry->second;
                   return Eigen::Vector3s(lidar2reference.leftCols<3>() * lidar_point + lidar2reference.col(3));
                 });
}

} // namespace rko_lio::core

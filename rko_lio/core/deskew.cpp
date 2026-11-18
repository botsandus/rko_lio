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
#include <sophus/se3.hpp>
#include <tsl/robin_map.h>

namespace rko_lio::core {
void deskew_scan(Vector3sVector& scan, const Timestamps& timestamps, const MotionPrior& motion) {
  const Sophus::SE3s start2reference = relative_pose_at_time(motion, timestamps.max).inverse();
  // keyed by the stamp in ns; prime bucket counts, so regularly spaced stamps spread over the buckets
  tsl::robin_pg_map<Nsec::rep, Eigen::Matrix<Scalar, 3, 4>> reference_T_stamp;
  std::transform(scan.cbegin(), scan.cend(), timestamps.per_point.cbegin(), scan.begin(),
                 [&](const Eigen::Vector3s& point, const Nsec timestamp) {
                   const auto [it, inserted] = reference_T_stamp.try_emplace(timestamp.count());
                   if (inserted) {
                     it.value() = (start2reference * relative_pose_at_time(motion, timestamp)).matrix3x4();
                   }
                   const Eigen::Matrix<Scalar, 3, 4>& reference_T_point = it->second;
                   return Eigen::Vector3s(reference_T_point.leftCols<3>() * point + reference_T_point.col(3));
                 });
}

} // namespace rko_lio::core

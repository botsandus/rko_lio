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

/**
 * @file deskew.hpp
 * Contains the declaration for scan deskewing.
 */

#pragma once
#include "util.hpp"

namespace rko_lio::core {

/**
 * Deskew a scan in place, from the lidar frame to the base frame at reference_time.
 *
 * @param timestamps Absolute per-point timestamps.
 * @param motion Motion prior of the base frame, spanning the scan.
 * @param per_stamp Compose one transform per distinct stamp and apply it to that stamp's points, for lidars whose
 * points share exact stamps; otherwise each point is transformed on its own.
 */
void deskew_scan(Vector3sVector& scan,
                 const Timestamps& timestamps,
                 const Nsec reference_time,
                 const MotionPrior& motion,
                 const Sophus::SE3s& lidar2base,
                 const bool per_stamp);

} // namespace rko_lio::core

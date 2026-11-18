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
 * @file lio.hpp
 * Core LIO class and utilities for RKO-LIO.
 */

#pragma once
#include "process_timestamps.hpp"
#include "util.hpp"
#include "voxel_hash_map.hpp"

#include <tbb/task_arena.h>

#include <deque>
#include <span>

/** Core namespace containing LIO data structures and state definitions. */
namespace rko_lio::core {
/** Core LiDAR-inertial odometry algorithm class. */
class LIO {
public:
  /** Configuration parameters for odometry. */
  struct Config {
    /** Enable scan deskewing. */
    bool deskew = true;

    /** Maximum number of ICP iterations. */
    size_t max_iterations = 100;

    /** Size of voxel grid (m). */
    Scalar voxel_size = 1.0;

    /** Max points per voxel. */

    /** Maximum lidar range (m). */
    Scalar max_range = 100.0;

    /** Minimum lidar range (m). */
    Scalar min_range = 1.0;

    /** ICP convergence threshold. */
    Scalar convergence_criterion = 1e-5;

    /** Max distance for correspondences (m). */
    Scalar max_correspondence_distance = 0.5;

    /** Thread count for data association (0 = automatic). */
    int max_num_threads = 0;

    /** Enable initialization phase. */
    bool initialization_phase = false;

    /** Enable double downsampling. */
    bool double_downsample = true;

    /** Restrict the pose to x, y and yaw. */
    bool flat = false;
  };

  /** Configuration parameters. */
  Config config;

  /** Local map. */
  VoxelHashMap map;

  /** Bounds ICP parallelism to max_num_threads without touching any other TBB user in the process. */
  tbb::task_arena arena;

  /** Current LiDAR state estimate. */
  State lidar_state;

  /** IMU bias estimates when initialization is enabled. */
  ImuBias imu_bias;

  /** IMU measurement statistics since last LiDAR frame. */
  IntervalStats interval_stats;

  /** How many times `reset()` has been called. */
  std::size_t reset_count = 0;
  /** What the last scan given to `register_scan` had; its ICP fields stay zero if no ICP ran. */
  ScanStats scan_stats;

  explicit LIO(const Config& config_);

  /** Add an IMU measurement expressed in the base frame. */
  void add_imu_measurement(const ImuControl& base_imu);

  /**
   * Add an IMU measurement expressed in the IMU frame and transform it
   * to the base frame using the given extrinsic calibration.
   * @param extrinsic_imu2base Extrinsic transform from IMU to base frame.
   * @param raw_imu Raw IMU measurement.
   */
  void add_imu_measurement(const Sophus::SE3s& extrinsic_imu2base, const ImuControl& raw_imu);

  /**
   * Register a LiDAR scan, applying deskewing based on the initial motion guess
   * and clipping points beyond valid range.
   * @param scan Input raw point cloud.
   * @param timestamps Absolute per-point timestamps, plus the scan's own min/max, as already
   *   computed by process_timestamps.
   * @return Deskewed and clipped point cloud.
   */
  Vector3sVector register_scan(Vector3sVector scan, const Timestamps& timestamps);

  /**
   * Register one scan given as several clouds, e.g. one per phase-locked lidar, without concatenating them.
   * @param scan The clouds, in the base frame.
   * @param timestamps One entry per cloud: its per-point timestamps; every entry carries the whole scan's min/max.
   * @return The deskewed and clipped clouds; empty if the scan was skipped.
   */
  std::vector<Vector3sVector> register_scan(std::vector<Vector3sVector> scan, std::span<const Timestamps> timestamps);

  /**
   * Register a scan whose clouds are in the lidar frame: the extrinsic calibration from lidar to base is applied first.
   * @param extrinsic_lidar2base Extrinsic from lidar to base frame.
   * @param scan The clouds, as for register_scan above.
   * @param timestamps As for register_scan above.
   * @return The deskewed and clipped clouds, in the base frame; empty if the scan was skipped.
   */
  std::vector<Vector3sVector> register_scan(const Sophus::SE3s& extrinsic_lidar2base,
                                            std::vector<Vector3sVector> scan,
                                            std::span<const Timestamps> timestamps);

  /** Restart, treating the next scan as if it were the first. Keeps the IMU bias estimates; forces
   *  `config.initialization_phase` off. */
  void restart();

  /** How many registered poses `poses_with_timestamps` keeps; the oldest goes first. */
  static constexpr std::size_t POSE_WINDOW_SIZE = 600;

  /** The last `POSE_WINDOW_SIZE` registered scan poses with their timestamps. A caller that keeps the trajectory moves
   *  them out as they come. */
  std::deque<std::pair<Nsec, Sophus::SE3s>> poses_with_timestamps;

  /** Base-frame state propagated from IMU measurements. Runs ahead of `lidar_state` between scans; reset to the
   *  optimized lidar pose after each successful registration. Used for gravity compensation and for publishing
   *  IMU-rate odometry from the ROS wrapper.
   */
  State imu_state;

private:
  /**
   * Initialize internal odometry state using the given lidar timestamp.
   * @param lidar_time Current lidar timestamp.
   */
  void initialize(const Nsec lidar_time);

  /** First-scan path: stamps state, optionally seeds the map, logs the pose. */
  std::vector<Vector3sVector> bootstrap_first_scan(std::vector<Vector3sVector> scan, const Nsec current_lidar_time);

  /** Motion prior for the coming scan, from the IMU interval average */
  MotionPrior motion_prior_from_imu(const Nsec current_lidar_time);

  /** Appends the lidar state's time and pose to `poses_with_timestamps`, dropping its oldest beyond the window. */
  void record_pose();

  /** True if odometry initialization has been completed. */
  bool initialized_ = false;

  ScanFragmentCheck fragment_check_;

  /** Timestamp of the most recent real IMU measurement. */
  Nsec last_real_imu_time_{0};

  /** Angular velocity of last true IMU measurement expressed in base frame. */
  Eigen::Vector3s last_real_base_imu_ang_vel_ = Eigen::Vector3s::Zero();
};
} // namespace rko_lio::core

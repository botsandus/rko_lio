#pragma once
#include <Eigen/Dense>
#include <vector>

#include "lio.hpp"

namespace rko_lio::core {

struct PreprocessingResult {
  std::vector<Vector3sVector> filtered_clouds; // one entry per input cloud
  Vector3sVector keypoints;
  Vector3sVector map_points; // populated only when config.double_downsample; empty otherwise
};

// clip the input clouds, then downsample them as one
PreprocessingResult preprocess_payload(std::vector<Vector3sVector> scan, const LIO::Config& config);

} // namespace rko_lio::core

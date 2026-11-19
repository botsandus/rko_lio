#include "preprocess_payload.hpp"
#include "voxel_down_sample.hpp"

namespace rko_lio::core {

PreprocessingResult preprocess_payload(std::vector<Vector3sVector> clouds, const LIO::Config& config) {
  for (Vector3sVector& cloud : clouds) {
    std::erase_if(cloud, [&](const auto& point) {
      const Scalar point_range = point.norm();
      return point_range <= config.min_range || point_range >= config.max_range;
    });
  }

  if (config.double_downsample) {
    Vector3sVector downsampled = voxel_down_sample(clouds, config.voxel_size * static_cast<Scalar>(0.5));
    Vector3sVector keypoints = voxel_down_sample(downsampled, config.voxel_size * static_cast<Scalar>(1.5));
    return {
        .filtered_clouds = std::move(clouds),
        .keypoints = std::move(keypoints),
        .map_points = std::move(downsampled),
    };
  }
  Vector3sVector keypoints = voxel_down_sample(clouds, config.voxel_size);
  return {.filtered_clouds = std::move(clouds), .keypoints = std::move(keypoints), .map_points = {}};
}

} // namespace rko_lio::core

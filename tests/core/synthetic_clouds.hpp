#pragma once
#include "rko_lio/core/lio.hpp"
#include <Eigen/Core>
#include <array>
#include <span>
#include <utility>
#include <vector>

namespace rko_lio::tests {

// Hollow cube of side 2*half_extent centered at the origin, six faces sampled at
// `spacing`.
inline std::vector<Eigen::Vector3s> make_hollow_cube(double spacing = 0.5, double half_extent = 5.0) {
  std::vector<Eigen::Vector3s> points;
  const int n = static_cast<int>(2 * half_extent / spacing) + 1;
  points.reserve(static_cast<size_t>(6 * n * n));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const double a = -half_extent + i * spacing;
      const double b = -half_extent + j * spacing;
      points.emplace_back(half_extent, a, b);
      points.emplace_back(-half_extent, a, b);
      points.emplace_back(a, half_extent, b);
      points.emplace_back(a, -half_extent, b);
      points.emplace_back(a, b, half_extent);
      points.emplace_back(a, b, -half_extent);
    }
  }
  return points;
}

// register_scan of one cloud already in the base frame
inline rko_lio::core::Vector3sVector register_cloud(rko_lio::core::LIO& lio,
                                                    rko_lio::core::Vector3sVector cloud,
                                                    const rko_lio::core::Timestamps& timestamps) {
  std::vector<rko_lio::core::Vector3sVector> scan;
  scan.push_back(std::move(cloud));
  const std::array<Sophus::SE3s, 1> identity{};
  std::vector<rko_lio::core::Vector3sVector> registered =
      lio.register_scan(std::move(scan), std::span(&timestamps, 1), identity);
  return registered.empty() ? rko_lio::core::Vector3sVector{} : std::move(registered.front());
}

} // namespace rko_lio::tests

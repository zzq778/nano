#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace rescue_vision {

// All coordinates refer to the supplied BGR image, never ground coordinates.
enum class TopFaceKind { BrightSurface, IsolatedColorFallback };

struct TopFace {
  cv::Point2d center;
  cv::Rect component;
  cv::Point min_pixel;
  cv::Point max_pixel;  // Inclusive bounds, matching observed pixels.
  int pixels = 0;
  bool split = false;
  TopFaceKind kind = TopFaceKind::BrightSurface;
};

// Reproduces the selected r3 thresholds. No camera, detector, count prior,
// global state, file IO, or runtime backend selection belongs to this kernel.
std::vector<TopFace> detectTopFaces(const cv::Mat& bgr);

}  // namespace rescue_vision

#ifndef RV_ORANGE_NATIVE_HPP
#define RV_ORANGE_NATIVE_HPP

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace rv {

// Port of the V10 "orange" branch (color_core.py / estimate). Nothing else.
struct OrangeResult {
  std::string status = "reject", reason;
  bool has_bbox = false, has_center = false, has_rims = false;
  cv::Rect bbox;
  int mask_area_px = 0, n_profiles = 0;
  cv::Point2d center_local;
  double top_depth_fraction = 0, edge_contrast = 0, scatter_px = 0, survival = 0;
  cv::Vec2d back_line, front_line;  // (slope, intercept) from polyfit degree 1
  std::vector<cv::Point> back_points, front_points;
  cv::Mat mask;
};

// bgr_roi must be a non-empty CV_8UC3 BGR ROI. Throws std::invalid_argument on
// inputs Python's cv2 pipeline would also reject (empty / wrong type / channels).
OrangeResult estimate_orange(const cv::Mat& bgr_roi);

}  // namespace rv

#endif  // RV_ORANGE_NATIVE_HPP

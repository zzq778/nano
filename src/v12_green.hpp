#pragma once

// V12 exploratory image-space refinement (green only).
// Performance port of the reference Python refine_green / refine_candidates
// group veto. No detector, no camera model, no object count.

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <vector>

namespace v12 {

struct Seed {
    double x, y, x0, x1, y0, y1;
};

struct Config {
    double back_expand_width = .4;
    double back_upper_width = .2;
    double min_depth_width = .16;
    double max_depth_width = .70;
    double minimum_contrast = 7;
    double fit_residual_fraction = .035;
    int minimum_profiles = 7;
    double minimum_survival = .60;
};

struct Result {
    cv::Point2d initial_center, center, proposed_center;
    bool has_proposed = false;
    std::string status = "fallback", reason = "insufficient_two_rim_evidence";
    int n_profiles = 0, conflicting_index = -1;
    bool has_lines = false;
    cv::Vec2d back_line, front_line, median_contrast;
    double rear_y = 0, front_y = 0, survival = 0;
    std::vector<cv::Point2d> back_points, front_points, debug_back, debug_front;
};

// Engineering session API: preserves local scoring while allowing near/color scheduling.
struct PreparedGreen { cv::Mat mask, lum; };
PreparedGreen prepare_green(const cv::Mat& bgr, bool gaussian3=false);
Result refine_green_prepared(const PreparedGreen& frame, const Seed& seed, const Config& config);

// One HSV mask / one 7x7 L blur for the whole group. gaussian3 adds an extra
// BGR 3x3 sigma=0.8 blur before everything else.
std::vector<Result> refine_green_group(const cv::Mat& bgr,
                                       const std::vector<Seed>& seeds,
                                       const Config& config,
                                       bool gaussian3 = false);

} // namespace v12

// Faithful C++14/OpenCV4 port of the orange branch of color_core.py.
// Only orange is implemented; black/blue branches are intentionally skipped.
// The ROI is a dynamic cv::Mat: no knowledge of image position or count.

#include "orange_native.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace rv {
namespace {

constexpr int kMinProfiles = 6;  // Python: < 6 / < 6

// np.round(x, ndigits): round-half-to-even (banker's rounding).
// std::nearbyint honours the default FE_TONEAREST mode, i.e. ties-to-even,
// so scaling by 10^ndigits reproduces Python's even rounding.
double round_py(double value, int ndigits) {
  if (!std::isfinite(value)) return value;
  const double scale = std::pow(10.0, static_cast<double>(ndigits));
  return std::nearbyint(value * scale) / scale;
}

// np.median: sorted; odd -> middle, even -> mean of the two middle samples.
double median_py(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const std::size_t n = values.size();
  if (n % 2 == 1) return values[n / 2];
  return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

// np.polyfit(x, y, 1): least squares y = slope * x + intercept in double.
// NumPy promotes the (float32) inputs with "+ 0.0" to float64 and solves the
// same linear least-squares problem; the centered normal equations below are
// the numerically stable form of it.
void polyfit1(const std::vector<double>& xs, const std::vector<double>& ys,
              double& slope, double& intercept) {
  const std::size_t n = xs.size();  // every call site guarantees n >= 6
  double sx = 0.0, sy = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sx += xs[i];
    sy += ys[i];
  }
  const double mx = sx / static_cast<double>(n);
  const double my = sy / static_cast<double>(n);
  double sxx = 0.0, sxy = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double dx = xs[i] - mx;
    sxx += dx * dx;
    sxy += dx * (ys[i] - my);
  }
  if (sxx == 0.0) {  // unreachable: profile abscissae increase by step >= 3
    slope = 0.0;
    intercept = my;
    return;
  }
  slope = sxy / sxx;
  intercept = my - slope * mx;
}

// comp_roi(roi, 'orange'): returns the 0/255 component mask and the int bbox
// (x, y, w, h) of the largest 8-connected component, or an empty mask.
cv::Mat orange_component(const cv::Mat& roi, cv::Rect& bbox) {
  cv::Mat hsv;
  cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

  // raw = cv2.inRange(hsv, (0,25,20), (23,255,255))
  cv::Mat raw;
  cv::inRange(hsv, cv::Scalar(0, 25, 20), cv::Scalar(23, 255, 255), raw);
  // if color=='orange': raw |= cv2.inRange(hsv, (173,35,20), (179,255,255))
  cv::Mat raw_wrap;
  cv::inRange(hsv, cv::Scalar(173, 35, 20), cv::Scalar(179, 255, 255), raw_wrap);
  cv::bitwise_or(raw, raw_wrap, raw);

  // np.ones((3,3), np.uint8) kernel, MORPH_OPEN
  const cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
  cv::morphologyEx(raw, raw, cv::MORPH_OPEN, kernel);

  cv::Mat labels, stats, centroids;
  const int n =
      cv::connectedComponentsWithStats(raw, labels, stats, centroids, 8, CV_32S);
  if (n < 2) return cv::Mat();  // Python: None

  // i = 1 + np.argmax(s[1:, 4])  -> first label on area ties (strict >).
  int best = 1;
  for (int i = 2; i < n; ++i) {
    if (stats.at<int>(i, cv::CC_STAT_AREA) > stats.at<int>(best, cv::CC_STAT_AREA)) {
      best = i;
    }
  }

  bbox = cv::Rect(stats.at<int>(best, cv::CC_STAT_LEFT),
                  stats.at<int>(best, cv::CC_STAT_TOP),
                  stats.at<int>(best, cv::CC_STAT_WIDTH),
                  stats.at<int>(best, cv::CC_STAT_HEIGHT));

  // mask = np.uint8(lab == i) * 255  -> 0 / 255 CV_8U.
  cv::Mat mask;
  cv::inRange(labels, cv::Scalar(static_cast<double>(best)),
              cv::Scalar(static_cast<double>(best)), mask);
  return mask;
}

}  // namespace

OrangeResult estimate_orange(const cv::Mat& bgr_roi) {
  // Python's cv2 pipeline requires an 8-bit 3-channel BGR image.
  if (bgr_roi.empty()) {
    throw std::invalid_argument("estimate_orange: empty ROI");
  }
  if (bgr_roi.depth() != CV_8U || bgr_roi.channels() != 3) {
    throw std::invalid_argument("estimate_orange: ROI must be CV_8UC3 BGR");
  }

  OrangeResult out;

  // mask, bb = comp_roi(roi, 'orange')
  cv::Rect bbox;
  cv::Mat mask = orange_component(bgr_roi, bbox);
  if (mask.empty()) {
    out.status = "reject";
    out.reason = "no color component";
    return out;
  }

  const int x = bbox.x;
  const int y = bbox.y;
  const int w = bbox.width;
  const int h = bbox.height;
  const int H = mask.rows;

  // basic = {'bbox_local': [x,y,w,h], 'mask_area_px': countNonZero(mask)}
  out.has_bbox = true;
  out.bbox = bbox;
  out.mask_area_px = cv::countNonZero(mask);
  out.mask = mask;  // keep the mask for verification, as in Python

  if (w < 25 || h < 25) {
    out.status = "reject";
    out.reason = "too small component";
    return out;
  }

  const double hd = static_cast<double>(h);
  const double wd = static_cast<double>(w);

  // hsv = cv2.cvtColor(roi, COLOR_BGR2HSV); lab = ... (lab is blue-only here)
  cv::Mat hsv;
  cv::cvtColor(bgr_roi, hsv, cv::COLOR_BGR2HSV);

  // sm = cv2.GaussianBlur(hsv[:,:,1], (5,7), 1.4).astype(np.float32)
  // Blur on the 8U channel first, then convert to float32 - Python order.
  cv::Mat sat8;
  cv::extractChannel(hsv, sat8, 1);
  cv::Mat sat_blur8;
  cv::GaussianBlur(sat8, sat_blur8, cv::Size(5, 7), 1.4);
  cv::Mat grad;  // grad = sm for orange
  sat_blur8.convertTo(grad, CV_32F);

  // d = max(3, min(8, round(h*.02)))
  const int d = std::max(3, std::min(8, static_cast<int>(std::nearbyint(hd * 0.02))));
  // step = max(3, round(w/40))
  const int step = std::max(3, static_cast<int>(std::nearbyint(wd / 40.0)));

  std::vector<cv::Point> bpts, fpts;
  std::vector<double> strength;

  // for xx in range(x+int(.25*w), x+int(.75*w), step)
  const int x_begin = x + static_cast<int>(0.25 * wd);
  const int x_end = x + static_cast<int>(0.75 * wd);
  const int row_band_hi = y + static_cast<int>(0.7 * hd);
  const int lo_band_y = y + static_cast<int>(0.1 * hd);
  const int hi_band_y = y + static_cast<int>(0.6 * hd);

  for (int xx = x_begin; xx < x_end; xx += step) {
    // row = np.where(mask[:,xx] > 0)[0]; row = row[row >= y & row <= y+int(.7*h)]
    // back = int(row[0])
    int back = -1;
    for (int r = 0; r < H; ++r) {
      if (mask.at<uchar>(r, xx) > 0 && r >= y && r <= row_band_hi) {
        back = r;
        break;
      }
    }
    if (back < 0) continue;

    // lo = max(d, y+int(.1*h), back+int(.09*h))
    // hi = min(H-d-1, y+int(.6*h), back+int(.62*h))
    const int lo =
        std::max(d, std::max(lo_band_y, back + static_cast<int>(0.09 * hd)));
    const int hi = std::min(
        H - d - 1, std::min(hi_band_y, back + static_cast<int>(0.62 * hd)));
    if (hi <= lo) continue;

    // cand = np.arange(lo, hi); measure = grad[cand+d,xx] - grad[cand-d,xx]
    // grad is float32 in NumPy, so the differences are float32 too.
    const int n_cand = hi - lo;
    std::vector<float> measure(static_cast<std::size_t>(n_cand));
    for (int k = 0; k < n_cand; ++k) {
      const int r = lo + k;
      measure[static_cast<std::size_t>(k)] =
          grad.at<float>(r + d, xx) - grad.at<float>(r - d, xx);
    }
    // j = int(np.argmax(measure)) -> first maximum wins.
    int j = 0;
    for (int k = 1; k < n_cand; ++k) {
      if (measure[static_cast<std::size_t>(k)] >
          measure[static_cast<std::size_t>(j)]) {
        j = k;
      }
    }
    const int front = lo + j;
    const double val = static_cast<double>(measure[static_cast<std::size_t>(j)]);
    const double span = static_cast<double>(front - back);

    // if val > 8 and .07*h < front-back < .58*h
    if (val > 8.0 && span > 0.07 * hd && span < 0.58 * hd) {
      bpts.emplace_back(xx, back);
      fpts.emplace_back(xx, front);
      strength.push_back(val);
    }
  }

  const int n_before = static_cast<int>(fpts.size());
  if (n_before < kMinProfiles) {
    out.status = "reject";
    out.reason = "insufficient signed two-rim profiles";
    out.n_profiles = n_before;
    return out;
  }

  // bp/fp are float32 arrays in Python; the values are integers, so casting
  // through float keeps the exact same numbers.
  std::vector<double> fpx(n_before), fpy(n_before);
  std::vector<double> bpx(n_before), bpy(n_before);
  for (int i = 0; i < n_before; ++i) {
    fpx[static_cast<std::size_t>(i)] =
        static_cast<double>(static_cast<float>(fpts[static_cast<std::size_t>(i)].x));
    fpy[static_cast<std::size_t>(i)] =
        static_cast<double>(static_cast<float>(fpts[static_cast<std::size_t>(i)].y));
    bpx[static_cast<std::size_t>(i)] =
        static_cast<double>(static_cast<float>(bpts[static_cast<std::size_t>(i)].x));
    bpy[static_cast<std::size_t>(i)] =
        static_cast<double>(static_cast<float>(bpts[static_cast<std::size_t>(i)].y));
  }

  // fit = np.polyfit(fp[:,0], fp[:,1], 1)
  double fit_slope = 0.0, fit_intercept = 0.0;
  polyfit1(fpx, fpy, fit_slope, fit_intercept);

  // resid = abs(fp[:,1] - polyval(fit, fp[:,0]))
  std::vector<double> resid(n_before);
  for (int i = 0; i < n_before; ++i) {
    const std::size_t k = static_cast<std::size_t>(i);
    resid[k] = std::fabs(fpy[k] - (fit_slope * fpx[k] + fit_intercept));
  }

  // keep = resid < max(8, .06*h)
  const double keep_thr = std::max(8.0, 0.06 * hd);
  std::vector<char> keep(n_before, 0);
  int ksum = 0;
  for (int i = 0; i < n_before; ++i) {
    const std::size_t k = static_cast<std::size_t>(i);
    if (resid[k] < keep_thr) {
      keep[k] = 1;
      ++ksum;
    }
  }
  if (ksum < kMinProfiles) {
    out.status = "reject";
    out.reason = "inconsistent front rim";
    out.n_profiles = n_before;
    return out;
  }

  std::vector<double> fkx, fky, bkx, bky;
  fkx.reserve(static_cast<std::size_t>(ksum));
  fky.reserve(static_cast<std::size_t>(ksum));
  bkx.reserve(static_cast<std::size_t>(ksum));
  bky.reserve(static_cast<std::size_t>(ksum));
  for (int i = 0; i < n_before; ++i) {
    const std::size_t k = static_cast<std::size_t>(i);
    if (!keep[k]) continue;
    fkx.push_back(fpx[k]);
    fky.push_back(fpy[k]);
    bkx.push_back(bpx[k]);
    bky.push_back(bpy[k]);
  }

  // fl = polyfit(fp[keep]); bl = polyfit(bp[keep])
  double fl_slope = 0.0, fl_intercept = 0.0;
  double bl_slope = 0.0, bl_intercept = 0.0;
  polyfit1(fkx, fky, fl_slope, fl_intercept);
  polyfit1(bkx, bky, bl_slope, bl_intercept);

  // cx = x + .5*w; re = polyval(bl, cx); fr = polyval(fl, cx); depth = fr - re
  const double cx = static_cast<double>(x) + 0.5 * wd;
  const double re = bl_slope * cx + bl_intercept;
  const double fr = fl_slope * cx + fl_intercept;
  const double depth = fr - re;

  // scatter = median(resid[keep]); contrast = median(st[keep])
  std::vector<double> resid_keep;
  std::vector<double> strength_keep;
  resid_keep.reserve(static_cast<std::size_t>(ksum));
  strength_keep.reserve(static_cast<std::size_t>(ksum));
  for (int i = 0; i < n_before; ++i) {
    const std::size_t k = static_cast<std::size_t>(i);
    if (!keep[k]) continue;
    resid_keep.push_back(resid[k]);
    strength_keep.push_back(strength[k]);
  }
  const double scatter = median_py(resid_keep);
  const double contrast = median_py(strength_keep);
  const double survival =
      static_cast<double>(ksum) / static_cast<double>(n_before);
  const double frac = depth / hd;

  // reason list, joined with "; " in this exact order.
  std::vector<std::string> reasons;
  if (y <= 2) reasons.push_back("ROI clips upper face");
  if (frac < 0.10 || frac > 0.58) reasons.push_back("implausible depth fraction");
  if (contrast < 18.0) reasons.push_back("weak front-edge contrast");
  if (scatter > std::max(8.0, 0.05 * hd)) reasons.push_back("incoherent front-edge line");
  if (survival < 0.7) reasons.push_back("unstable across horizontal samples");

  out.status = reasons.empty() ? "two_rim_candidate" : "weak_or_reject";
  if (reasons.empty()) {
    out.reason = "two signed rims accepted";
  } else {
    out.reason.clear();
    for (std::size_t i = 0; i < reasons.size(); ++i) {
      if (i != 0) out.reason += "; ";
      out.reason += reasons[i];
    }
  }

  out.has_center = true;
  out.has_rims = true;
  out.center_local =
      cv::Point2d(round_py(cx, 1), round_py(0.5 * (re + fr), 1));
  out.top_depth_fraction = round_py(frac, 3);
  out.edge_contrast = round_py(contrast, 1);
  out.scatter_px = round_py(scatter, 1);
  out.survival = round_py(survival, 3);
  out.n_profiles = ksum;

  out.back_line = cv::Vec2d(bl_slope, bl_intercept);
  out.front_line = cv::Vec2d(fl_slope, fl_intercept);

  // bp[keep].astype(int) / fp[keep].astype(int): truncation toward zero.
  out.back_points.reserve(static_cast<std::size_t>(ksum));
  out.front_points.reserve(static_cast<std::size_t>(ksum));
  for (int i = 0; i < n_before; ++i) {
    const std::size_t k = static_cast<std::size_t>(i);
    if (!keep[k]) continue;
    out.back_points.emplace_back(static_cast<int>(std::trunc(bpx[k])),
                                 static_cast<int>(std::trunc(bpy[k])));
    out.front_points.emplace_back(static_cast<int>(std::trunc(fpx[k])),
                                  static_cast<int>(std::trunc(fpy[k])));
  }

  return out;
}

}  // namespace rv

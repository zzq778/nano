#include <algorithm>
#include <array>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <rescue_vision/topface.hpp>
#include <stdexcept>
#include <utility>

namespace rescue_vision {
namespace {
struct Component {
  int label;
  cv::Rect bounds;
  int area;
  double cy;
};

double roundTenth(double value) {
  // Preserve the original binary value when locating the decimal halfway point.
  // Multiplication in double can round e.g. 51.15 up to an artificial exact
  // tie.
  return static_cast<double>(
      std::nearbyint(static_cast<long double>(value) * 10.L) / 10.L);
}

double percentileSorted(const std::vector<double>& values, double fraction) {
  if (values.empty()) throw std::invalid_argument("empty percentile input");
  const double index = fraction * (values.size() - 1);
  const auto lower = static_cast<size_t>(index);
  const auto upper = std::min(lower + 1, values.size() - 1);
  return values[lower] + (values[upper] - values[lower]) * (index - lower);
}

double percentile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  return percentileSorted(values, fraction);
}

template <typename Histogram>
double histogramPercentile(const Histogram& counts, size_t total,
                           double fraction, int offset = 0) {
  if (total == 0) throw std::invalid_argument("empty histogram");
  const double position = fraction * (total - 1);
  const size_t lower = static_cast<size_t>(position);
  const size_t upper = std::min(lower + 1, total - 1);
  size_t cumulative = 0;
  int low_value = -1;
  for (size_t i = 0; i < counts.size(); ++i) {
    cumulative += counts[i];
    if (low_value < 0 && cumulative > lower) low_value = static_cast<int>(i);
    if (cumulative > upper)
      return offset + low_value +
             (static_cast<int>(i) - low_value) * (position - lower);
  }
  throw std::logic_error("histogram population mismatch");
}

std::vector<float> smooth(const std::vector<float>& values) {
  if (values.empty()) return {};
  cv::Mat input(1, static_cast<int>(values.size()), CV_32F,
                const_cast<float*>(values.data()));
  cv::Mat output;
  cv::GaussianBlur(input, output, {}, 4.0);
  return std::vector<float>(output.ptr<float>(),
                            output.ptr<float>() + output.cols);
}

std::vector<std::pair<int, double>> peaks(const std::vector<float>& values,
                                          int minimum_distance) {
  const auto signal = smooth(values);
  if (signal.size() < 11) return {};
  const int radius = std::max(32, static_cast<int>(minimum_distance * 1.5));
  std::vector<std::pair<int, double>> candidates;
  for (int i = 5; i < static_cast<int>(signal.size()) - 5; ++i) {
    if (!(signal[i] > signal[i - 1] && signal[i] >= signal[i + 1])) continue;
    const auto left = std::min_element(signal.begin() + std::max(0, i - radius),
                                       signal.begin() + i);
    const auto right = std::min_element(
        signal.begin() + i + 1,
        signal.begin() + std::min<int>(signal.size(), i + radius));
    const double prominence = signal[i] - std::max(*left, *right);
    if (prominence >= 1.7) candidates.emplace_back(i, prominence);
  }
  std::stable_sort(
      candidates.begin(), candidates.end(),
      [](const auto& a, const auto& b) { return a.second > b.second; });
  std::vector<std::pair<int, double>> kept;
  for (const auto& candidate : candidates) {
    const bool separated =
        std::all_of(kept.begin(), kept.end(), [&](const auto& other) {
          return std::abs(candidate.first - other.first) >= minimum_distance;
        });
    if (separated) kept.push_back(candidate);
  }
  std::sort(kept.begin(), kept.end());
  return kept;
}

TopFace summarize(const std::vector<cv::Point>& points,
                  const cv::Rect& component, bool split) {
  TopFace result;
  result.component = component;
  result.pixels = static_cast<int>(points.size());
  result.split = split;
  result.min_pixel = result.max_pixel = points.front();
  double sx = 0, sy = 0;
  for (const auto& p : points) {
    sx += p.x;
    sy += p.y;
    result.min_pixel.x = std::min(result.min_pixel.x, p.x);
    result.min_pixel.y = std::min(result.min_pixel.y, p.y);
    result.max_pixel.x = std::max(result.max_pixel.x, p.x);
    result.max_pixel.y = std::max(result.max_pixel.y, p.y);
  }
  result.center = {roundTenth(sx / points.size()),
                   roundTenth(sy / points.size())};
  return result;
}

std::vector<TopFace> splitFaces(const cv::Mat& luminance, const cv::Mat& green,
                                const cv::Mat& hsv, double level) {
  cv::Mat blurred, edges, bright, labels, stats, centers;
  cv::GaussianBlur(luminance, blurred, {3, 3}, 0);
  cv::Canny(blurred, edges, 10, 30);
  cv::compare(luminance, level, bright, cv::CMP_GT);
  cv::bitwise_and(bright, green, bright);
  cv::morphologyEx(bright, bright, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8U));
  cv::dilate(edges, edges, cv::Mat::ones(2, 2, CV_8U));
  bright.setTo(0, edges);
  const int count =
      cv::connectedComponentsWithStats(bright, labels, stats, centers, 8);
  std::vector<Component> components;
  std::vector<double> areas;
  for (int i = 1; i < count; ++i) {
    const int* s = stats.ptr<int>(i);
    const cv::Rect r(s[0], s[1], s[2], s[3]);
    if (s[4] < 225 || r.width < 16 || r.height < 8 ||
        double(s[4]) / r.area() < .14)
      continue;
    components.push_back({i, r, s[4], centers.at<double>(i, 1)});
    areas.push_back(s[4]);
  }
  if (components.empty()) return {};
  const double median_area = percentile(areas, .5);
  components.erase(
      std::remove_if(
          components.begin(), components.end(),
          [&](const Component& c) {
            const auto& r = c.bounds;
            const int cx = r.x + r.width / 2;
            const int y0 = std::min(
                green.rows, r.y + r.height + std::max(7, int(r.height * .06)));
            const int y1 =
                std::min(green.rows, y0 + std::max(16, int(r.height * .23)));
            const int x0 = std::max(0, cx - std::max(8, int(r.width * .18)));
            const int x1 =
                std::min(green.cols, cx + std::max(8, int(r.width * .18)));
            const cv::Rect below_bounds(x0, y0, x1 - x0, y1 - y0);
            const double below =
                below_bounds.area() > 0
                    ? double(cv::countNonZero(green(below_bounds))) /
                          below_bounds.area()
                    : 0;
            if (below > .2 || c.area > median_area * 1.7) return false;
            if (below_bounds.empty() || c.area < 600) return true;
            cv::Mat blue;
            cv::inRange(hsv(below_bounds), cv::Scalar(100, 110, 18),
                        cv::Scalar(140, 255, 255), blue);
            return cv::countNonZero(blue) <= .60 * below_bounds.area();
          }),
      components.end());
  if (components.empty()) return {};
  std::vector<Component> scale_seeds;
  for (const auto& c : components)
    if (c.bounds.height < 115 && c.bounds.width < 320 && c.area > 1600)
      scale_seeds.push_back(c);
  if (scale_seeds.empty()) {
    scale_seeds = components;
    std::stable_sort(
        scale_seeds.begin(), scale_seeds.end(),
        [](const auto& a, const auto& b) { return a.area < b.area; });
    scale_seeds.resize(std::max<size_t>(1, scale_seeds.size() / 2));
  }
  std::vector<double> widths, heights, ys;
  for (const auto& c : scale_seeds) {
    widths.push_back(c.bounds.width);
    heights.push_back(c.bounds.height);
    ys.push_back(c.cy);
  }
  const double base_width = percentile(widths, .5),
               base_height = percentile(heights, .5),
               base_y = percentile(ys, .5);
  const auto size_at = [&](double y) {
    const double factor =
        std::max(.65, std::min(1.7, (y + 200) / (base_y + 200)));
    return cv::Size2d(base_width * factor, base_height * factor);
  };
  cv::Mat smooth_lum, gx, gy;
  // Gradients only contribute when the corresponding split test is reached.
  const auto prepareGradient = [&](cv::Mat& gradient, int dx, int dy) {
    if (!gradient.empty()) return;
    if (smooth_lum.empty()) {
      cv::Mat lum_float;
      luminance.convertTo(lum_float, CV_32F);
      cv::GaussianBlur(lum_float, smooth_lum, {5, 5}, 0);
    }
    cv::Sobel(smooth_lum, gradient, CV_32F, dx, dy, 3);
  };
  std::vector<TopFace> result;
  for (const auto& component : components) {
    const auto& r = component.bounds;
    const auto expected = size_at(component.cy);
    int rows = std::max(
        1,
        std::min(
            3, int(std::floor(r.height / std::max(40., expected.height) * 1.22 +
                              .35))));
    if (component.area < 2500) rows = 1;
    std::vector<int> row_splits;
    if (rows > 1) {
      prepareGradient(gy, 0, 1);
      std::vector<float> profile(r.height, 0);
      for (int y = 0; y < r.height; ++y) {
        float sum = 0;
        int pixels = 0;
        for (int x = r.x; x < r.x + r.width; ++x)
          if (green.at<uchar>(r.y + y, x)) {
            sum += std::max(0.f, -gy.at<float>(r.y + y, x));
            ++pixels;
          }
        profile[y] = sum / std::max(1, pixels);
      }
      profile = smooth(profile);
      auto candidates =
          peaks(profile, std::max(22, int(.36 * expected.height)));
      for (int j = 1; j < rows; ++j) {
        const double target =
            rows == 2 ? std::min(r.height * .8, 1.25 * expected.height)
                      : double(r.height) * j / rows;
        std::pair<double, int> best(-1, -1);
        for (const auto& peak : candidates) {
          const int k = peak.first;
          if (!(k > r.height * .15 && k < r.height * .87)) continue;
          const double z = (k - target) / std::max(17., expected.height * .34);
          best = std::max(
              best,
              std::make_pair(
                  double(profile[k]) * (.18 + .82 * std::exp(-.5 * z * z)), k));
        }
        if (best.second >= 0 &&
            std::all_of(row_splits.begin(), row_splits.end(), [&](int k) {
              return std::abs(best.second - k) >
                     std::max(24, int(.32 * expected.height));
            }))
          row_splits.push_back(best.second);
      }
      std::sort(row_splits.begin(), row_splits.end());
    }
    std::vector<int> cuts{0};
    cuts.insert(cuts.end(), row_splits.begin(), row_splits.end());
    cuts.push_back(r.height);
    for (size_t band = 1; band < cuts.size(); ++band) {
      const int begin = cuts[band - 1], end = cuts[band];
      if (end - begin < 8) continue;
      std::vector<cv::Point> points;
      std::vector<size_t> column_counts(r.width, 0);
      for (int y = r.y + begin; y < r.y + end; ++y)
        for (int x = r.x; x < r.x + r.width; ++x)
          if (labels.at<int>(y, x) == component.label) {
            points.emplace_back(x, y);
            ++column_counts[x - r.x];
          }
      if (points.size() < 180) continue;
      const int xmin = int(histogramPercentile(column_counts, points.size(),
                                               .02, r.x)),
                xmax = int(histogramPercentile(column_counts, points.size(),
                                               .98, r.x)),
                width = xmax - xmin;
      const auto local_size = size_at(r.y + (begin + end) / 2.);
      std::vector<int> xcuts;
      if (width / std::max(local_size.width, 1.) > 1.7 &&
          end - begin < 1.3 * local_size.height) {
        prepareGradient(gx, 1, 0);
        std::vector<float> profile(width + 1, 0);
        for (int x = xmin; x <= xmax; ++x) {
          float sum = 0;
          int pixels = 0;
          for (int y = r.y + begin; y < r.y + end; ++y)
            if (green.at<uchar>(y, x)) {
              sum += std::abs(gx.at<float>(y, x));
              ++pixels;
            }
          profile[x - xmin] = sum / std::max(1, pixels);
        }
        profile = smooth(profile);
        std::pair<double, int> best(-1, -1);
        for (const auto& peak : peaks(profile, 20))
          if (peak.first > .2 * width && peak.first < .8 * width)
            best =
                std::max(best, std::make_pair(peak.second, peak.first + xmin));
        if (best.second >= 0) xcuts.push_back(best.second);
      }
      std::vector<int> boundaries{xmin - 1};
      boundaries.insert(boundaries.end(), xcuts.begin(), xcuts.end());
      boundaries.push_back(xmax + 1);
      for (size_t i = 1; i < boundaries.size(); ++i) {
        std::vector<cv::Point> selected;
        for (const auto& p : points)
          if (p.x >= boundaries[i - 1] && p.x < boundaries[i])
            selected.push_back(p);
        if (selected.size() >= 140)
          result.push_back(
              summarize(selected, r, !row_splits.empty() || !xcuts.empty()));
      }
    }
  }
  return result;
}
}  // namespace

std::vector<TopFace> detectTopFaces(const cv::Mat& bgr) {
  if (bgr.empty() || bgr.type() != CV_8UC3)
    throw std::invalid_argument("topface requires nonempty uint8 BGR image");
  cv::Mat hsv, lab, lum, green;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
  cv::extractChannel(lab, lum, 0);
  cv::inRange(hsv, cv::Scalar(35, 60, 25), cv::Scalar(83, 255, 255), green);
  if (cv::countNonZero(green) < 100) return {};
  std::array<size_t, 256> histogram{};
  size_t population = 0;
  for (int y = 0; y < green.rows; ++y) {
    const auto* mask_row = green.ptr<uchar>(y);
    const auto* lum_row = lum.ptr<uchar>(y);
    for (int x = 0; x < green.cols; ++x) {
      if (mask_row[x]) {
        ++histogram[lum_row[x]];
        ++population;
      }
    }
  }
  const double q20 = histogramPercentile(histogram, population, .2),
               q80 = histogramPercentile(histogram, population, .8);
  auto proposals =
      splitFaces(lum, green, hsv, std::nearbyint(q20 + .69 * (q80 - q20)));
  cv::Mat opened, labels, stats, centers;
  cv::morphologyEx(green, opened, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8U));
  const int count =
      cv::connectedComponentsWithStats(opened, labels, stats, centers, 8);
  for (int i = 1; i < count; ++i) {
    const int* s = stats.ptr<int>(i);
    if (s[4] < 300 || s[2] < 12 || s[3] < 12 ||
        double(s[4]) / (s[2] * s[3]) < .35 ||
        std::max(double(s[2]) / s[3], double(s[3]) / s[2]) > 2.3)
      continue;
    const bool explained =
        std::any_of(proposals.begin(), proposals.end(), [&](const TopFace& p) {
          const int x = int(std::nearbyint(p.center.x)),
                    y = int(std::nearbyint(p.center.y));
          return x >= 0 && x < labels.cols && y >= 0 && y < labels.rows &&
                 labels.at<int>(y, x) == i;
        });
    if (explained) continue;
    TopFace p;
    p.component = {s[0], s[1], s[2], s[3]};
    p.pixels = s[4];
    p.kind = TopFaceKind::IsolatedColorFallback;
    p.center = {roundTenth(centers.at<double>(i, 0)),
                roundTenth(centers.at<double>(i, 1))};
    p.min_pixel = {s[0], s[1]};
    p.max_pixel = {s[0] + s[2] - 1, s[1] + s[3] - 1};
    proposals.push_back(p);
  }
  return proposals;
}
}  // namespace rescue_vision

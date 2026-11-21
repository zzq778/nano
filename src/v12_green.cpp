#include "v12_green.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <utility>

namespace v12 {
namespace {

// ---------------------------------------------------------------------------
// Small double-precision helpers mirroring the NumPy/Python primitives.
// ---------------------------------------------------------------------------

struct Triple { double x, y, s; };   // candidate [x, y, strength]
struct Pair   { double x, y; };      // [x, y]
struct Line   { double slope = 0.0, intercept = 0.0; };

inline bool maskAt(const cv::Mat& mask, int y, int x) {
    return mask.at<unsigned char>(y, x) != 0;
}

inline double lval(const cv::Mat& lum, int y, int x) {
    return lum.at<double>(y, x);
}

inline double polyval(const Line& line, double x) {
    return line.slope * x + line.intercept;
}

// Deterministic 1st-order least squares, [slope, intercept] like np.polyfit.
// The rank-deficient branch reproduces lstsq's minimum-norm solution used by
// np.polyfit when the Vandermonde matrix is singular (<=> all x identical).
void polyfit1(const std::vector<double>& xs, const std::vector<double>& ys, Line& out) {
    const std::size_t n = xs.size();
    if (n == 0) {
        out.slope = 0.0;
        out.intercept = 0.0;
        return;
    }
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = xs[i];
        const double y = ys[i];
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    const double dn = static_cast<double>(n);
    const double denom = dn * sxx - sx * sx;
    if (denom != 0.0) {
        out.slope = (dn * sxy - sx * sy) / denom;
        out.intercept = (sy - out.slope * sx) / dn;
        return;
    }
    const double c = xs[0];
    const double meanY = sy / dn;
    out.slope = c * meanY / (c * c + 1.0);
    out.intercept = meanY / (c * c + 1.0);
}

// np.median semantics (average of the two middle samples for even counts).
double medianCopy(std::vector<double> values) {
    if (values.empty()) return 0.0;
    const std::size_t n = values.size();
    std::sort(values.begin(), values.end());
    if (n % 2 == 1) return values[n / 2];
    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

// Python fit(): polyfit, at most 3 residual-driven refits, final residuals are
// evaluated against every input point with the final line.
Line fitRobustLine(const std::vector<Pair>& p, double width, const Config& config,
                   std::vector<double>& residuals) {
    const std::size_t n = p.size();
    std::vector<double> xs(n), ys(n);
    for (std::size_t i = 0; i < n; ++i) {
        xs[i] = p[i].x;
        ys[i] = p[i].y;
    }
    Line line;
    polyfit1(xs, ys, line);
    const double tol = std::max(4.0, width * config.fit_residual_fraction);
    for (int iter = 0; iter < 3; ++iter) {
        std::vector<double> kx, ky;
        kx.reserve(n);
        ky.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double err = std::fabs(polyval(line, xs[i]) - ys[i]);
            if (err < tol) {
                kx.push_back(xs[i]);
                ky.push_back(ys[i]);
            }
        }
        if (kx.size() < 4) break;
        polyfit1(kx, ky, line);
    }
    residuals.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        residuals[i] = std::fabs(polyval(line, xs[i]) - ys[i]);
    }
    return line;
}

struct Coherent {
    bool ok = false;
    Line line;
    std::vector<Triple> chosen;   // one candidate per unique x column
};

// Python coherent_line(): 53-slope deterministic grid, subsampled intercept
// hypotheses, per-column voting (any hit per column), capped strength bonus,
// first argmax tie, then argmin per column, then a straight polyfit.
Coherent coherentLine(const std::vector<Triple>& points, double width) {
    Coherent result;
    const std::size_t n = points.size();
    if (n < 7) return result;

    // np.unique(x, return_inverse=True): sorted unique x plus per-point column.
    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&points](std::size_t a, std::size_t b) {
        return points[a].x < points[b].x;
    });
    std::vector<double> uniqueX;
    uniqueX.reserve(n);
    std::vector<int> column(n, 0);
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t idx = order[k];
        if (uniqueX.empty() || points[idx].x != uniqueX.back()) {
            uniqueX.push_back(points[idx].x);
        }
        column[idx] = static_cast<int>(uniqueX.size()) - 1;
    }
    const int ncol = static_cast<int>(uniqueX.size());

    const double tol = std::max(3.0, width * 0.025);
    const int hstep = std::max(1, static_cast<int>(n) / 70);
    const double delta = 1.3 / 52.0;   // np.linspace(-.65, .65, 53)

    std::vector<double> intercepts(n, 0.0);
    std::vector<double> hypotheses;
    hypotheses.reserve(n);
    std::vector<char> good(n, 0);
    std::vector<char> columnHit(static_cast<std::size_t>(ncol), 0);
    std::vector<char> bestGood;
    std::vector<char> slopeBestGood;

    bool haveBest = false;
    double bestScore = 0.0;
    double bestSlope = 0.0;
    double bestIntercept = 0.0;

    for (int si = 0; si < 53; ++si) {
        const double slope = (si == 52) ? 0.65 : (-0.65 + static_cast<double>(si) * delta);
        for (std::size_t i = 0; i < n; ++i) {
            intercepts[i] = points[i].y - slope * points[i].x;
        }
        hypotheses.clear();
        for (std::size_t i = 0; i < n; i += static_cast<std::size_t>(hstep)) {
            hypotheses.push_back(intercepts[i]);
        }
        const int m = static_cast<int>(hypotheses.size());

        int bestJ = 0;
        double bestJScore = 0.0;
        slopeBestGood.clear();

        for (int j = 0; j < m; ++j) {
            const double hyp = hypotheses[static_cast<std::size_t>(j)];
            std::fill(good.begin(), good.end(), 0);
            std::fill(columnHit.begin(), columnHit.end(), 0);
            int columnCount = 0;
            double strength = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                if (std::fabs(intercepts[i] - hyp) < tol) {
                    good[i] = 1;
                    strength += std::min(points[i].s, 40.0);
                    const std::size_t c = static_cast<std::size_t>(column[i]);
                    if (!columnHit[c]) {
                        columnHit[c] = 1;
                        ++columnCount;
                    }
                }
            }
            const double score = static_cast<double>(columnCount) + 0.002 * strength;
            if (j == 0 || score > bestJScore) {
                bestJScore = score;
                bestJ = j;
                slopeBestGood = good;
            }
        }

        if (!haveBest || bestJScore > bestScore) {
            bestScore = bestJScore;
            bestSlope = slope;
            bestIntercept = hypotheses[static_cast<std::size_t>(bestJ)];
            bestGood = slopeBestGood;
            haveBest = true;
        }
    }

    if (!haveBest || bestGood.empty()) return result;

    // One representative per unique x: argmin |y - (slope*x + intercept)|.
    for (int ci = 0; ci < ncol; ++ci) {
        const double x = uniqueX[static_cast<std::size_t>(ci)];
        const double target = bestSlope * x + bestIntercept;
        int sel = -1;
        double bestErr = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            if (!bestGood[i] || column[i] != ci) continue;
            const double err = std::fabs(points[i].y - target);
            if (sel < 0 || err < bestErr) {
                sel = static_cast<int>(i);
                bestErr = err;
            }
        }
        if (sel >= 0) result.chosen.push_back(points[static_cast<std::size_t>(sel)]);
    }

    result.ok = true;
    std::vector<double> xs(result.chosen.size()), ys(result.chosen.size());
    for (std::size_t i = 0; i < result.chosen.size(); ++i) {
        xs[i] = result.chosen[i].x;
        ys[i] = result.chosen[i].y;
    }
    polyfit1(xs, ys, result.line);
    return result;
}

// ---------------------------------------------------------------------------
// refine_green(image, seed) for one supplied span, on a precomputed group mask.
// ---------------------------------------------------------------------------

Result refineOne(const cv::Mat& mask, const cv::Mat& lum, const Seed& seed,
                 const Config& config, int rows, int cols) {
    Result out;
    out.initial_center = cv::Point2d(seed.x, seed.y);
    out.center = out.initial_center;

    const double width = seed.x1 - seed.x0;
    if (!(width > 0.0)) return out;   // degenerate span: keep fallback, no crash

    const int d = std::max(2, static_cast<int>(std::nearbyint(width * 0.018)));
    const int step = std::max(2, static_cast<int>(width / 45.0));
    const int xBegin = static_cast<int>(seed.x0 + 0.30 * width);
    const int xEnd = static_cast<int>(seed.x1 - 0.30 * width);
    const int suppressRadius = std::max(4, static_cast<int>(width * 0.025));

    std::vector<Triple> backCandidates;

    for (int x = xBegin; x < xEnd; x += step) {
        if (x < 0 || x >= cols) continue;
        const int lo = std::max(d, static_cast<int>(seed.y0 - config.back_expand_width * width));
        const int hi = std::min(std::min(rows - d - 1,
                                         static_cast<int>(seed.y0 + config.back_upper_width * width)),
                                static_cast<int>(seed.y));
        const int ny = hi - lo;
        if (ny < 3) continue;

        std::vector<double> score(static_cast<std::size_t>(ny), 0.0);
        for (int k = 0; k < ny; ++k) {
            const int y = lo + k;
            const double rise = lval(lum, y + d, x) - lval(lum, y - d, x);
            const bool mUp = maskAt(mask, y + d, x);
            const bool mDown = maskAt(mask, y - d, x);
            const bool entry = mUp && !mDown;
            const bool nearFlag = std::fabs(static_cast<double>(y) - seed.y0) < 0.22 * width;
            const double s1 = mUp ? rise : -100.0;
            const double s2 = (entry && nearFlag) ? std::fabs(rise) : -100.0;
            score[static_cast<std::size_t>(k)] = std::max(s1, s2);
        }

        // Up to four peaks per column, strict argmax (first wins ties).
        for (int peak = 0; peak < 4; ++peak) {
            int bi = 0;
            double bs = score[0];
            for (int k = 1; k < ny; ++k) {
                if (score[static_cast<std::size_t>(k)] > bs) {
                    bs = score[static_cast<std::size_t>(k)];
                    bi = k;
                }
            }
            const int by = lo + bi;
            if (bs < config.minimum_contrast) break;
            Triple t;
            t.x = static_cast<double>(x);
            t.y = static_cast<double>(by);
            t.s = bs;
            backCandidates.push_back(t);
            for (int k = 0; k < ny; ++k) {
                if (std::abs((lo + k) - by) < suppressRadius) {
                    score[static_cast<std::size_t>(k)] = -100.0;
                }
            }
        }
    }

    const Coherent model = coherentLine(backCandidates, width);

    std::vector<Pair> backs, fronts;
    std::vector<std::array<double, 2>> strengths;

    if (model.ok) {
        for (std::size_t ci = 0; ci < model.chosen.size(); ++ci) {
            const Triple& t = model.chosen[ci];
            const int x = static_cast<int>(t.x);
            if (x < 0 || x >= cols) continue;
            const int by = static_cast<int>(
                std::nearbyint(polyval(model.line, static_cast<double>(x))));
            const int low = std::max(d, by + static_cast<int>(config.min_depth_width * width));
            const int high = std::min(std::min(rows - d - 1,
                                               by + static_cast<int>(config.max_depth_width * width)),
                                      static_cast<int>(seed.y1 + 0.20 * width));
            const int nys = high - low;
            if (nys < 3) continue;

            std::vector<double> fall(static_cast<std::size_t>(nys), 0.0);
            for (int k = 0; k < nys; ++k) {
                const int y = low + k;
                const double v = lval(lum, y - d, x) - lval(lum, y + d, x);
                fall[static_cast<std::size_t>(k)] = maskAt(mask, y - d, x) ? v : -100.0;
            }
            int fi = 0;
            double fs = fall[0];
            for (int k = 1; k < nys; ++k) {
                if (fall[static_cast<std::size_t>(k)] > fs) {
                    fs = fall[static_cast<std::size_t>(k)];
                    fi = k;
                }
            }
            const int fy = low + fi;
            if (fs >= config.minimum_contrast) {
                Pair b;
                b.x = static_cast<double>(x);
                b.y = static_cast<double>(by);
                Pair f;
                f.x = static_cast<double>(x);
                f.y = static_cast<double>(fy);
                backs.push_back(b);
                fronts.push_back(f);
                strengths.push_back({t.s, fs});
            }
        }
    }

    out.n_profiles = static_cast<int>(backs.size());
    if (backs.empty() || static_cast<int>(backs.size()) < config.minimum_profiles) {
        return out;
    }

    std::vector<double> backErr, frontErr;
    fitRobustLine(backs, width, config, backErr);
    fitRobustLine(fronts, width, config, frontErr);
    const double tol = std::max(4.0, width * config.fit_residual_fraction);

    std::vector<char> keep(backs.size(), 0);
    int keepCount = 0;
    for (std::size_t i = 0; i < backs.size(); ++i) {
        if (backErr[i] < tol && frontErr[i] < tol) {
            keep[i] = 1;
            ++keepCount;
        }
    }
    const double survival = static_cast<double>(keepCount) / static_cast<double>(backs.size());

    if (keepCount < config.minimum_profiles || survival < config.minimum_survival) {
        out.reason = "inconsistent_rims";
        out.debug_back.reserve(backs.size());
        out.debug_front.reserve(fronts.size());
        for (const Pair& p : backs) out.debug_back.push_back(cv::Point2d(p.x, p.y));
        for (const Pair& p : fronts) out.debug_front.push_back(cv::Point2d(p.x, p.y));
        return out;
    }

    std::vector<Pair> keptBacks, keptFronts;
    std::vector<std::array<double, 2>> keptStrengths;
    std::vector<double> keptX;
    keptBacks.reserve(static_cast<std::size_t>(keepCount));
    keptFronts.reserve(static_cast<std::size_t>(keepCount));
    keptStrengths.reserve(static_cast<std::size_t>(keepCount));
    keptX.reserve(static_cast<std::size_t>(keepCount));

    std::vector<double> kbx, kby, kfx, kfy;
    kbx.reserve(static_cast<std::size_t>(keepCount));
    kby.reserve(static_cast<std::size_t>(keepCount));
    kfx.reserve(static_cast<std::size_t>(keepCount));
    kfy.reserve(static_cast<std::size_t>(keepCount));

    for (std::size_t i = 0; i < backs.size(); ++i) {
        if (!keep[i]) continue;
        keptBacks.push_back(backs[i]);
        keptFronts.push_back(fronts[i]);
        keptStrengths.push_back(strengths[i]);
        keptX.push_back(backs[i].x);
        kbx.push_back(backs[i].x);
        kby.push_back(backs[i].y);
        kfx.push_back(fronts[i].x);
        kfy.push_back(fronts[i].y);
    }

    // Final lines are refit on the surviving profiles (left/right must agree).
    Line bl;
    Line fl;
    polyfit1(kbx, kby, bl);
    polyfit1(kfx, kfy, fl);

    // Scan-support anchor: median supported x, not a brightness-weighted centre.
    const double cx = medianCopy(keptX);
    const double rearY = polyval(bl, cx);
    const double frontY = polyval(fl, cx);
    const double cy = 0.5 * (rearY + frontY);
    const double separation = frontY - rearY;

    if (!(0.12 * width < separation && separation < 0.75 * width)) {
        out.reason = "implausible_rim_separation";
        return out;
    }

    out.center = cv::Point2d(cx, cy);
    out.status = "edge_candidate";
    out.reason = "two_coherent_signed_rims";
    out.has_lines = true;
    out.back_line = cv::Vec2d(bl.slope, bl.intercept);
    out.front_line = cv::Vec2d(fl.slope, fl.intercept);
    out.rear_y = rearY;
    out.front_y = frontY;
    out.n_profiles = keepCount;
    out.survival = survival;

    std::vector<double> contrast0, contrast1;
    contrast0.reserve(static_cast<std::size_t>(keepCount));
    contrast1.reserve(static_cast<std::size_t>(keepCount));
    for (std::size_t i = 0; i < keptStrengths.size(); ++i) {
        contrast0.push_back(keptStrengths[i][0]);
        contrast1.push_back(keptStrengths[i][1]);
    }
    out.median_contrast = cv::Vec2d(medianCopy(contrast0), medianCopy(contrast1));

    out.back_points.reserve(keptBacks.size());
    out.front_points.reserve(keptFronts.size());
    for (const Pair& p : keptBacks) out.back_points.push_back(cv::Point2d(p.x, p.y));
    for (const Pair& p : keptFronts) out.front_points.push_back(cv::Point2d(p.x, p.y));

    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Group entrypoint: one HSV mask + one L blur for all seeds, then the
// cross-candidate consistency veto against the frozen local-centre anchors.
// Candidate count never enters the local optimisation.
// ---------------------------------------------------------------------------

PreparedGreen prepare_green(const cv::Mat& bgr, bool gaussian3) {
    cv::Mat work;
    if(gaussian3) cv::GaussianBlur(bgr,work,cv::Size(3,3),.8,.8);
    else work=bgr;
    PreparedGreen out; cv::Mat hsv,lab,l8;
    cv::cvtColor(work,hsv,cv::COLOR_BGR2HSV);
    cv::inRange(hsv,cv::Scalar(35,42,25),cv::Scalar(88,255,255),out.mask);
    cv::cvtColor(work,lab,cv::COLOR_BGR2Lab);
    cv::extractChannel(lab,l8,0);
    cv::GaussianBlur(l8,l8,cv::Size(7,7),0.,0.);
    l8.convertTo(out.lum,CV_64F);
    return out;
}

Result refine_green_prepared(const PreparedGreen& frame,const Seed& seed,const Config& config) {
    return refineOne(frame.mask,frame.lum,seed,config,frame.mask.rows,frame.mask.cols);
}

std::vector<Result> refine_green_group(const cv::Mat& bgr,
                                       const std::vector<Seed>& seeds,
                                       const Config& config,
                                       bool gaussian3) {
    std::vector<Result> results;
    results.reserve(seeds.size());

    if (seeds.empty()) return results;

    if (bgr.empty() || bgr.type() != CV_8UC3) {
        for (const Seed& s : seeds) {
            Result r;
            r.initial_center = cv::Point2d(s.x, s.y);
            r.center = r.initial_center;
            results.push_back(r);
        }
        return results;
    }

    const int rows = bgr.rows;
    const int cols = bgr.cols;

    cv::Mat work;
    if (gaussian3) {
        cv::GaussianBlur(bgr, work, cv::Size(3, 3), 0.8, 0.8);
    } else {
        work = bgr;
    }

    // (35,42,25)..(88,255,255) on HSV 8U, -> bool mask.
    cv::Mat hsv;
    cv::cvtColor(work, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(35, 42, 25), cv::Scalar(88, 255, 255), mask);

    // L channel: 7x7 GaussianBlur in 8U first, then float (matches astype(float)).
    cv::Mat lab;
    cv::cvtColor(work, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> labChannels;
    cv::split(lab, labChannels);
    cv::Mat l8, lum;
    cv::GaussianBlur(labChannels[0], l8, cv::Size(7, 7), 0.0, 0.0);
    l8.convertTo(lum, CV_64F);

    for (const Seed& s : seeds) {
        results.push_back(refineOne(mask, lum, s, config, rows, cols));
    }

    // Frozen snapshot of every local centre: a veto never rewrites these.
    std::vector<cv::Point2d> anchors(results.size());
    for (std::size_t i = 0; i < results.size(); ++i) anchors[i] = results[i].center;

    for (std::size_t i = 0; i < results.size(); ++i) {
        if (results[i].status != "edge_candidate") continue;
        const double xa = seeds[i].x0;
        const double xb = seeds[i].x1;
        const double margin = 0.08 * (xb - xa);
        for (std::size_t j = 0; j < anchors.size(); ++j) {
            if (i == j) continue;
            const double x = anchors[j].x;
            const double y = anchors[j].y;
            if (!(xa + margin < x && x < xb - margin)) continue;
            const double back = results[i].back_line[0] * x + results[i].back_line[1];
            const double front = results[i].front_line[0] * x + results[i].front_line[1];
            const double span = front - back;
            if (back + 0.10 * span < y && y < front - 0.10 * span) {
                results[i].proposed_center = results[i].center;
                results[i].has_proposed = true;
                results[i].center = results[i].initial_center;
                results[i].status = "fallback";
                results[i].reason = "rims_span_another_candidate";
                results[i].conflicting_index = static_cast<int>(j);
                break;
            }
        }
    }

    return results;
}

} // namespace v12

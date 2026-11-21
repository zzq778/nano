// 模块职责：已知比赛尺寸的地面模型拟合：比较当前颜色轮廓和投影，保留可信多解，失败不假装成功。
#include <rescue_vision/model_color.hpp>
#include <rescue_vision/model_geometry.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace rescue_vision {
namespace {
constexpr float pi = 3.14159265358979323846F;
struct Vertex {
  float x, y, z;
};
float cross(Point2D a, Point2D b, Point2D c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}
std::vector<Point2D> convexHull(std::vector<Point2D> p) {
  std::sort(p.begin(), p.end(),
            [](Point2D a, Point2D b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });
  p.erase(std::unique(p.begin(), p.end(),
                      [](Point2D a, Point2D b) { return a.x == b.x && a.y == b.y; }),
          p.end());
  if (p.size() < 3) return {};
  std::vector<Point2D> h;
  h.reserve(p.size() * 2);
  for (auto q : p) {
    while (h.size() > 1 && cross(h[h.size() - 2], h.back(), q) <= 0) h.pop_back();
    h.push_back(q);
  }
  const auto n = h.size();
  for (int i = static_cast<int>(p.size()) - 2; i >= 0; --i) {
    while (h.size() > n && cross(h[h.size() - 2], h.back(), p[i]) <= 0) h.pop_back();
    h.push_back(p[i]);
  }
  h.pop_back();
  return h;
}
// 模型至多八个顶点，投影和凸包候选使用定长存储，避免每次假设分配堆内存。
struct SmallPolygon {
  std::array<Point2D, 16> points{};
  size_t count{0};
  size_t size() const { return count; }
  bool empty() const { return count == 0; }
  Point2D* begin() { return points.data(); }
  Point2D* end() { return begin() + count; }
  const Point2D* begin() const { return points.data(); }
  const Point2D* end() const { return begin() + count; }
  Point2D& operator[](size_t i) { return points[i]; }
  const Point2D& operator[](size_t i) const { return points[i]; }
  void push_back(Point2D p) { points[count++] = p; }
};
SmallPolygon convexHull(SmallPolygon p) {
  std::sort(p.begin(), p.end(),
            [](Point2D a, Point2D b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });
  p.count = static_cast<size_t>(
      std::unique(p.begin(), p.end(),
                  [](Point2D a, Point2D b) { return a.x == b.x && a.y == b.y; }) -
      p.begin());
  if (p.size() < 3) return {};
  SmallPolygon h;
  for (auto q : p) {
    while (h.size() > 1 && cross(h[h.size() - 2], h[h.size() - 1], q) <= 0) --h.count;
    h.push_back(q);
  }
  const auto n = h.size();
  for (int i = static_cast<int>(p.size()) - 2; i >= 0; --i) {
    while (h.size() > n && cross(h[h.size() - 2], h[h.size() - 1], p[i]) <= 0) --h.count;
    h.push_back(p[i]);
  }
  --h.count;
  return h;
}
// 先裁剪真实投影轮廓，再计算当前画面里的外接框。
// “完整模型的框与画面求交”会保留画面外底角的左右极值，斜边被截断时会算错可见宽度。
// 模型最多8个投影点，凸轮廓经过4条图像边界裁剪最多12点；全程使用定长栈存储。
SmallPolygon visibleProjection(SmallPolygon polygon, const CameraIntrinsics& k, bool ordered) {
  const float xmax = float(k.width - 1), ymax = float(k.height - 1);
  bool clipped = false;
  for (auto p : polygon)
    clipped |= p.x < 0 || p.x > xmax || p.y < 0 || p.y > ymax;
  if (!clipped) return polygon;  // 完整目标维持原快速路径，不增加凸包计算。
  if (!ordered) polygon = convexHull(polygon);
  for (unsigned side = 0; side < 4 && !polygon.empty(); ++side) {
    const auto distance = [&](Point2D p) -> double {
      return side == 0 ? p.x : side == 1 ? double(xmax) - p.x
                            : side == 2 ? p.y : double(ymax) - p.y;
    };
    SmallPolygon next;
    auto previous = polygon[polygon.size() - 1];
    double before = distance(previous);
    for (auto current : polygon) {
      const double after = distance(current);
      if ((before >= 0) != (after >= 0)) {
        const double t = before / (before - after);
        Point2D intersection{float(previous.x + t * (current.x - previous.x)),
                             float(previous.y + t * (current.y - previous.y))};
        // 消除浮点交点在当前边界上的舍入漂移，后续仍裁剪其他边界。
        if (side < 2) intersection.x = side == 0 ? 0.F : xmax;
        else intersection.y = side == 2 ? 0.F : ymax;
        next.push_back(intersection);
      }
      if (after >= 0) next.push_back(current);
      previous = current;
      before = after;
    }
    polygon = next;
  }
  return polygon.size() >= 3 ? polygon : SmallPolygon{};
}
bool inside(Point2D p, const SmallPolygon& h) {
  if (h.size() < 3) return false;
  for (size_t i = 0; i < h.size(); ++i)
    if (cross(h[i], h[(i + 1) % h.size()], p) < 0) return false;
  return true;
}
struct ModelVertices {
  std::array<Vertex, 8> points{};
  size_t count{0};
  const Vertex* begin() const { return points.data(); }
  const Vertex* end() const { return begin() + count; }
};
ModelVertices vertices(uint8_t cls) {
  ModelVertices model;
  if (cls == 1) {
    const float r = .04F / std::sqrt(3.F);
    model.points[0] = {r, 0, 0};
    model.points[1] = {-.5F * r, .02F, 0};
    model.points[2] = {-.5F * r, -.02F, 0};
    model.points[3] = {0, 0, .04F * std::sqrt(2.F / 3.F)};
    model.count = 4;
    return model;
  }
  for (float z : {0.F, .04F})
    for (float a : {-1.F, 1.F})
      for (float b : {-1.F, 1.F})
        model.points[model.count++] = {a * (cls == 2 ? .04F : .02F), b * .02F, z};
  return model;
}
struct Evidence {
  int x0 = 0, y0 = 0, w = 0, h = 0, step = 1;
  std::vector<uint8_t>& mask;
  std::vector<uint8_t>& hidden;
  std::vector<float>& distance;
  explicit Evidence(ModelFitScratch& s) : mask(s.mask), hidden(s.hidden), distance(s.distance) {}
  unsigned foreground = 0, boundary = 0;
  bool isHidden(float x, float y) const {
    const int u = static_cast<int>(std::lround((x - x0) / step));
    const int v = static_cast<int>(std::lround((y - y0) / step));
    return u >= 0 && u < w && v >= 0 && v < h && hidden[v * w + u];
  }
  // Edge evaluation used to round and index the same sample twice: once
  // for occlusion and again for distance. Keep the old bounds semantics (a
  // coordinate just outside the ROI can round into its border label), but
  // share the computed integer index for the common visible-pixel path.
  float visibleEdgeDistance(float x, float y, bool* occluded) const {
    const int u = static_cast<int>(std::lround((x - x0) / step));
    const int v = static_cast<int>(std::lround((y - y0) / step));
    if (u < 0 || u >= w || v < 0 || v >= h) {
      *occluded = false;
      return 20.F;
    }
    const int index = v * w + u;
    *occluded = hidden[index] != 0;
    if (*occluded) return 0.F;  // The caller discards this sample.
    if (x < x0 || y < y0 || x > x0 + (w - 1) * step || y > y0 + (h - 1) * step) return 20.F;
    return std::min(20.F, distance[index] * step);
  }
};
Evidence evidence(const Detection2D& d, const RgbFrameView& rgb, const ModelFitOptions& options,
                  ModelFitScratch& scratch) {
  Evidence e(scratch);
  e.x0 = std::max(0, static_cast<int>(std::floor(d.bbox.x_min)) - 5);
  e.y0 = std::max(0, static_cast<int>(std::floor(d.bbox.y_min)) - 5);
  const int x1 = std::min(rgb.width - 1, static_cast<int>(std::ceil(d.bbox.x_max)) + 5);
  const int y1 = std::min(rgb.height - 1, static_cast<int>(std::ceil(d.bbox.y_max)) + 5);
  e.step = std::max(1, static_cast<int>(std::ceil(std::max(x1 - e.x0 + 1, y1 - e.y0 + 1) / 96.F)));
  e.w = (x1 - e.x0) / e.step + 1;
  e.h = (y1 - e.y0) / e.step + 1;
  e.mask.resize(e.w * e.h);
  e.hidden.resize(e.w * e.h);
  e.distance.assign(e.w * e.h, 1000.F);
  for (int v = 0; v < e.h; ++v)
    for (int u = 0; u < e.w; ++u) {
      auto& value = e.mask[v * e.w + u];
      value = modelColorMatch(rgb.pixels + static_cast<size_t>(e.y0 + v * e.step) * rgb.stride +
                                  (e.x0 + u * e.step) * 3,
                              d.class_id);
      const int px = e.x0 + u * e.step, py = e.y0 + v * e.step;
      e.hidden[v * e.w + u] = 0;
      for (size_t i = 0; i < options.occluders.size(); ++i) {
        const auto& box = options.occluders[i];
        if (px < box.x_min || px > box.x_max || py < box.y_min || py > box.y_max) continue;
        const bool classified = options.occluder_classes.size() == options.occluders.size();
        if (!classified ||
            (options.occluder_classes[i] <= 3 &&
             modelColorMatch(rgb.pixels + static_cast<size_t>(py) * rgb.stride + px * 3,
                             options.occluder_classes[i]))) {
          e.hidden[v * e.w + u] = 1;
          break;
        }
      }
      if (px < d.bbox.x_min || px > d.bbox.x_max || py < d.bbox.y_min || py > d.bbox.y_max)
        value = 0;
      if (e.hidden[v * e.w + u]) value = 0;
    }
  // 紧外接框两侧连续延伸的细黑条可作为地面线证据；不能仅因区域水平就擦掉宽带或物块真实底部。
  if (d.class_id == 1 && e.w > 4) {
    int first = -1;
    const int max_rows = std::max(1, int(.10F * (d.bbox.y_max - d.bbox.y_min) / e.step));
    for (int v = 0; v <= e.h; ++v) {
      bool stripe =
          v < e.h &&
          modelColorMatch(
              rgb.pixels + static_cast<size_t>(e.y0 + v * e.step) * rgb.stride + e.x0 * 3, 1) &&
          modelColorMatch(rgb.pixels + static_cast<size_t>(e.y0 + v * e.step) * rgb.stride +
                              (e.x0 + (e.w - 1) * e.step) * 3,
                          1) &&
          e.x0 < d.bbox.x_min - 1 && e.x0 + (e.w - 1) * e.step > d.bbox.x_max + 1;
      // 两侧孤立黑点不足以证明地面线；按同一有界采样分辨率检查连续性。
      for (int u = 1; stripe && u < e.w - 1; ++u)
        stripe = modelColorMatch(rgb.pixels + static_cast<size_t>(e.y0 + v * e.step) * rgb.stride +
                                     (e.x0 + u * e.step) * 3,
                                 1);
      if (stripe && first < 0) first = v;
      if (!stripe && first >= 0) {
        if (v - first <= max_rows)
          for (int row = first; row < v; ++row)
            std::fill(e.mask.begin() + row * e.w, e.mask.begin() + (row + 1) * e.w, 0);
        first = -1;
      }
    }
  }
  // 只删除极小的断开颜色噪点，保留较大的分离区域，包括合并物块，供后续反例检查。
  scratch.labels.assign(e.w * e.h, -1);
  scratch.queue.clear();
  scratch.queue.reserve(e.w * e.h);
  unsigned largest = 0;
  int component = 0;
  for (int i = 0; i < e.w * e.h; ++i)
    if (e.mask[i] && scratch.labels[i] < 0) {
      const size_t start = scratch.queue.size();
      scratch.queue.push_back(i);
      scratch.labels[i] = component;
      for (size_t cursor = start; cursor < scratch.queue.size(); ++cursor) {
        const int q = scratch.queue[cursor], u = q % e.w, v = q / e.w;
        const int neighbors[4] = {u ? q - 1 : -1, u + 1 < e.w ? q + 1 : -1, v ? q - e.w : -1,
                                  v + 1 < e.h ? q + e.w : -1};
        for (int n : neighbors)
          if (n >= 0 && e.mask[n] && scratch.labels[n] < 0) {
            scratch.labels[n] = component;
            scratch.queue.push_back(n);
          }
      }
      largest = std::max(largest, static_cast<unsigned>(scratch.queue.size() - start));
      ++component;
    }
  // 临时借用距离缓冲保存连通分量计数，稍后会覆盖为距离变换结果。
  e.distance.assign(std::max(e.w * e.h, component), 0.F);
  for (int label : scratch.labels)
    if (label >= 0) ++e.distance[label];
  for (int i = 0; i < e.w * e.h; ++i) {
    if (e.mask[i] && e.distance[scratch.labels[i]] < std::min(4.F, float(largest) * .02F))
      e.mask[i] = 0;
    e.foreground += e.mask[i];
  }
  e.distance.assign(e.w * e.h, 1000.F);
  for (int v = 1; v < e.h - 1; ++v)
    for (int u = 1; u < e.w - 1; ++u) {
      const int i = v * e.w + u;
      if (e.mask[i] && !e.hidden[i - 1] && !e.hidden[i + 1] && !e.hidden[i - e.w] &&
          !e.hidden[i + e.w] &&
          (!e.mask[i - 1] || !e.mask[i + 1] || !e.mask[i - e.w] || !e.mask[i + e.w])) {
        e.distance[i] = 0;
        ++e.boundary;
      }
    }
  // 对观测轮廓做有界两遍近似距离变换，供模型边缘误差评价。
  for (int v = 0; v < e.h; ++v)
    for (int u = 0; u < e.w; ++u) {
      const int i = v * e.w + u;
      if (u) e.distance[i] = std::min(e.distance[i], e.distance[i - 1] + 1);
      if (v) e.distance[i] = std::min(e.distance[i], e.distance[i - e.w] + 1);
      if (u && v) e.distance[i] = std::min(e.distance[i], e.distance[i - e.w - 1] + 1.4142F);
    }
  for (int v = e.h - 1; v >= 0; --v)
    for (int u = e.w - 1; u >= 0; --u) {
      const int i = v * e.w + u;
      if (u + 1 < e.w) e.distance[i] = std::min(e.distance[i], e.distance[i + 1] + 1);
      if (v + 1 < e.h) e.distance[i] = std::min(e.distance[i], e.distance[i + e.w] + 1);
      if (u + 1 < e.w && v + 1 < e.h)
        e.distance[i] = std::min(e.distance[i], e.distance[i + e.w + 1] + 1.4142F);
    }
  return e;
}
struct Candidate {
  float x = 0, y = 0, yaw = 0, cost = 1e9F, bbox_error = 1e9F, edge_error = 1e9F, visible = 0;
};
// 单次拟合的评价器：投影、框误差、颜色轮廓误差和重叠率共用预算与临时证据。
struct Fitter {
  const CameraIntrinsics& k;
  const RigidTransform3D& tf;
  const Detection2D& d;
  const Evidence& e;
  const ModelVertices model;
  const ModelFitOptions& options;
  unsigned evaluations = 0;
  bool exhausted = false;
  ModelFitResult* diagnostics = nullptr;
  bool budget() {
    if (evaluations >= 6000 || std::chrono::steady_clock::now() >= options.deadline)
      exhausted = true;
    return exhausted;
  }
  SmallPolygon project(const Candidate& c, bool base = false, bool make_hull = true) const {
    SmallPolygon p;
    const float cosine = std::cos(c.yaw), sine = std::sin(c.yaw);
    for (auto a : model) {
      if (base && a.z != 0) continue;
      const float x = c.x + cosine * a.x - sine * a.y;
      const float y = c.y + sine * a.x + cosine * a.y;
      if (base) {
        p.push_back({x, y});
        continue;
      }
      const float dx = x - tf.translation[0], dy = y - tf.translation[1],
                  dz = a.z - tf.translation[2];
      const auto& r = tf.rotation;
      const float ox = r[0] * dx + r[3] * dy + r[6] * dz;
      const float oy = r[1] * dx + r[4] * dy + r[7] * dz;
      const float oz = r[2] * dx + r[5] * dy + r[8] * dz;
      if (oz <= .01F) return {};
      const Point2D q{k.fx * ox / oz + k.cx, k.fy * oy / oz + k.cy};
      if (!std::isfinite(q.x) || !std::isfinite(q.y)) return {};
      p.push_back(q);
    }
    return make_hull ? convexHull(std::move(p)) : p;
  }
  // 给定朝向后，用完整外接框的四条边求地面中心XY；它是初值，不是姿态接受证据。
  // 迭代更新极值顶点；四边均可见时才采用解析初值，遮挡/截断或病态时保留原搜索。
  bool initializeFromBox(Candidate* output) {
    Candidate current=*output;
    const double cosine=std::cos(current.yaw),sine=std::sin(current.yaw);
    const auto& r=tf.rotation;
    std::array<std::array<double,3>,8> offset{};
    size_t count=0;
    for(auto a:model){
      const double dx=cosine*a.x-sine*a.y-tf.translation[0];
      const double dy=sine*a.x+cosine*a.y-tf.translation[1],dz=a.z-tf.translation[2];
      offset[count++]={{r[0]*dx+r[3]*dy+r[6]*dz,
                        r[1]*dx+r[4]*dy+r[7]*dz,
                        r[2]*dx+r[5]*dy+r[8]*dz}};
    }
    // 四边依次为左、右、上、下；每次更新负责该边的三维极值顶点，最多迭代四轮。
    const double observed[4]={d.bbox.x_min,d.bbox.x_max,d.bbox.y_min,d.bbox.y_max};
    for(unsigned iteration=0;iteration<4;++iteration){
      if(budget())return false;
      ++evaluations;
      const auto pixels=project(current,false,false);
      if(pixels.size()!=count)return false;
      size_t extreme[4]={};
      for(size_t i=1;i<count;++i){
        if(pixels[i].x<pixels[extreme[0]].x)extreme[0]=i;
        if(pixels[i].x>pixels[extreme[1]].x)extreme[1]=i;
        if(pixels[i].y<pixels[extreme[2]].y)extreme[2]=i;
        if(pixels[i].y>pixels[extreme[3]].y)extreme[3]=i;
      }
      const double mx=.5*(pixels[extreme[0]].x+pixels[extreme[1]].x);
      const double my=.5*(pixels[extreme[2]].y+pixels[extreme[3]].y);
      double aa=0,ab=0,bb=0,ac=0,bc=0;unsigned rows=0;
      for(unsigned side=0;side<4;++side){
        const bool horizontal=side<2;
        const double limit=horizontal?k.width:k.height;
        if(observed[side]<2||observed[side]>limit-3)continue;
        const double u=horizontal?observed[side]:mx,v=horizontal?my:observed[side];
        if(e.isHidden(float(u),float(v)))continue;
        const double f=horizontal?k.fx:k.fy,principal=horizontal?k.cx:k.cy;
        const unsigned axis=horizontal?0:1;
        const double q=observed[side]-principal;
        const auto& o=offset[extreme[side]];
        // 针孔式 f*光学坐标 - (像素-主点)*光学深度=0，整理为 a*X+b*Y=value。
        const double a=f*r[axis]-q*r[2],b=f*r[axis+3]-q*r[5];
        const double value=q*o[2]-f*o[axis];
        aa+=a*a;ab+=a*b;bb+=b*b;ac+=a*value;bc+=b*value;++rows;
      }
      // 四条直线约束做二元最小二乘；病态、遮挡或截断一律回到有界搜索，不返回半成品。
      const double determinant=aa*bb-ab*ab,scale=aa+bb;
      if(rows<4||!std::isfinite(determinant)||determinant<=1e-10*scale*scale)return false;
      const double x=(ac*bb-bc*ab)/determinant,y=(bc*aa-ac*ab)/determinant;
      if(!std::isfinite(x)||!std::isfinite(y)||std::hypot(x-tf.translation[0],y-tf.translation[1])>4)return false;
      const double change=std::hypot(x-current.x,y-current.y);
      current.x=float(x);current.y=float(y);
      if(change<1e-6)break;
    }
    if(project(current,false,false).size()!=count)return false;
    *output=current;return true;
  }
  Candidate evaluate(Candidate c, bool edges) {
    if (exhausted || (evaluations % 16 == 0 && budget())) return Candidate{};
    ++evaluations;
    const auto p = visibleProjection(project(c, false, edges), k, edges);
    if (p.empty()) return Candidate{};
    BoundingBox2D box{1e9F, 1e9F, -1e9F, -1e9F};
    for (auto q : p) {
      box.x_min = std::min(box.x_min, q.x);
      box.y_min = std::min(box.y_min, q.y);
      box.x_max = std::max(box.x_max, q.x);
      box.y_max = std::max(box.y_max, q.y);
    }
    // 这里只比较真正可见轮廓的框，不把图像边缘当作实物底边。
    const float midx = (box.x_min + box.x_max) * .5F, midy = (box.y_min + box.y_max) * .5F;
    const float a = e.isHidden(box.x_min, midy) ? 0 : box.x_min - d.bbox.x_min;
    const float b = e.isHidden(box.x_max, midy) ? 0 : box.x_max - d.bbox.x_max;
    const float f = e.isHidden(midx, box.y_min) ? 0 : box.y_min - d.bbox.y_min;
    const float g = e.isHidden(midx, box.y_max) ? 0 : box.y_max - d.bbox.y_max;
    c.bbox_error = std::sqrt((a * a + b * b + f * f + g * g) / 4);
    c.edge_error = 0;
    if (edges) {
      unsigned samples = 0, visible = 0;
      for (size_t i = 0; i < p.size(); ++i) {
        auto u = p[i], v = p[(i + 1) % p.size()];
        for (int j = 0; j < 8; ++j) {
          float t = (j + .5F) / 8;
          const float x = u.x + (v.x - u.x) * t, y = u.y + (v.y - u.y) * t;
          if (x < 1 || x > k.width - 2 || y < 1 || y > k.height - 2) continue;
          ++samples;
          bool occluded = false;
          const float edge_distance = e.visibleEdgeDistance(x, y, &occluded);
          if (occluded) continue;
          c.edge_error += edge_distance;
          ++visible;
        }
      }
      c.visible = samples ? float(visible) / samples : 0;
      c.edge_error = visible >= 12 ? c.edge_error / visible : 1e9F;
    }
    c.cost = c.bbox_error + (edges ? 1.5F * c.edge_error : 0);
    if(edges && diagnostics && std::isfinite(c.cost) && c.cost<diagnostics->best_effort_cost &&
       c.bbox_error<1e8F && c.edge_error<1e8F && std::hypot(c.x-tf.translation[0],c.y-tf.translation[1])<4){
      diagnostics->best_effort={c.x,c.y,c.yaw,c.bbox_error,{}};
      diagnostics->best_effort_available=true;diagnostics->best_effort_cost=c.cost;
    }
    return c;
  }
  // 粗细两级坐标下降，步长逐层缩小；这里只搜索参数，不负责决定接受，最终仍走统一valid判定。
  Candidate refine(Candidate c, bool edges) {
    c = evaluate(c, edges);
    float xy = edges ? .004F : .025F, angle = .10F;
    for (int level = 0; level < 8; ++level) {
      const auto before = c;
      for (int axis = 0; axis < (edges ? 3 : 2); ++axis)
        for (int repeat = 0; repeat < 2; ++repeat) {
          Candidate best = c;
          for (float sign : {-1.F, 1.F}) {
            Candidate n = c;
            if (axis == 0)
              n.x += sign * xy;
            else if (axis == 1)
              n.y += sign * xy;
            else
              n.yaw += sign * angle;
            n = evaluate(n, edges);
            if (n.cost < best.cost) best = n;
          }
          // 两次重复使用相同步长；若第一次未改进，第二次会评价完全相同的两个点。
          // 仅跳过确定重复，不缩小方向、尺度或最终当前RGB验收集合。
          if (!(best.cost < c.cost)) break;
          c = best;
        }
      // 仅精搜索在步长降至亚毫米后允许提前收敛；粗XY搜索仍遍历各尺度，避免过早停在错误区域。
      if (edges && xy <= .00025F && angle <= .006F &&
          std::hypot(c.x - before.x, c.y - before.y) < 1e-6F &&
          std::abs(c.yaw - before.yaw) < 1e-5F && before.cost - c.cost < 1e-4F)
        break;
      xy *= .55F;
      angle *= .55F;
    }
    return c;
  }
  float overlap(const Candidate& c) const {
    const auto h = project(c);
    // Most pixels in the object ROI cannot lie inside the model polygon.
    // Cheap axis-aligned containment avoids a full edge-crossing loop there;
    // inside() is unchanged for all pixels that might actually be inside.
    float min_x = std::numeric_limits<float>::infinity();
    float min_y = std::numeric_limits<float>::infinity();
    float max_x = -min_x, max_y = -min_y;
    for (const auto& p : h) {
      min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
      min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
    }
    unsigned both = 0, either = 0;
    for (int v = 0; v < e.h; ++v) {
      const float y = float(e.y0 + v * e.step);
      const bool possible_y = y >= min_y && y <= max_y;
      for (int u = 0; u < e.w; ++u) {
        const int index = v * e.w + u;
        if (e.hidden[index]) continue;
        const bool a = e.mask[index];
        const float x = float(e.x0 + u * e.step);
        const bool b = possible_y && x >= min_x && x <= max_x && inside({x, y}, h);
        both += a && b;
        either += a || b;
      }
    }
    return either ? float(both) / either : 0;
  }
};
}  // 命名空间结束

ModelFitResult fitGroundModel(const Detection2D& d, const RgbFrameView& rgb,
                              const CameraIntrinsics& k, const RigidTransform3D& tf,
                              const GeometricObservation& coarse, const ModelFitOptions& options) {
  ModelFitResult result;
  if (d.class_id > 3 || !coarse.geometry_valid || !validateCameraIntrinsics(k) || !rgb.pixels ||
      rgb.width != k.width || rgb.height != k.height || rgb.width > 10000 || rgb.height > 10000 ||
      rgb.stride < rgb.width * 3 || !std::isfinite(coarse.pose_x) || !std::isfinite(coarse.pose_y))
    return result;
  for (auto x : tf.rotation)
    if (!std::isfinite(x)) return result;
  for (auto x : tf.translation)
    if (!std::isfinite(x)) return result;
  if (tf.translation[2] <= 0) return result;
  const auto& b = d.bbox;
  if (!std::isfinite(b.x_min) || !std::isfinite(b.x_max) || !std::isfinite(b.y_min) ||
      !std::isfinite(b.y_max))
    return result;
  if (b.x_min < 0 || b.y_min < 0 || b.x_max > rgb.width || b.y_max > rgb.height ||
      b.x_max - b.x_min < 8 || b.y_max - b.y_min < 8)
    return result;
  const int clipped =
      (b.x_min < 2) + (b.y_min < 2) + (b.x_max > rgb.width - 3) + (b.y_max > rgb.height - 3);
  if (clipped > 1) return result;
  const auto budgetResult = [&]() {
    result.status = ModelFitStatus::BUDGET;
    result.reason = uint8_t(ModelFitReason::BUDGET);
    result.candidates.clear();
    result.footprint.clear();
    return result;
  };
  if (std::chrono::steady_clock::now() >= options.deadline) return budgetResult();
  result.status = ModelFitStatus::REJECTED;
  result.reason = uint8_t(ModelFitReason::INSUFFICIENT_EVIDENCE);
  ModelFitScratch local_scratch;
  auto& scratch = options.scratch ? *options.scratch : local_scratch;
  const auto e = evidence(d, rgb, options, scratch);
  if (e.foreground < 20 || e.boundary < 12) return result;
  Fitter fitter{k, tf, d, e, vertices(d.class_id), options, 0, false};
  fitter.diagnostics=&result;
  const float period = d.class_id == 1 ? 2 * pi / 3 : (d.class_id == 2 ? pi : pi / 2);
  // 检测框是外接矩形，颜色边界还受像素采样、反光和分割影响；不再要求二者逐像素重合。
  // 640x480实拍中3~5px差异常见，但错误单块套群体可达数十像素，仍由轮廓/重叠联合拒绝。
  const float tolerance = std::max(5.F, float(e.step) * 1.5F);
  const auto normalize = [&](float a) {
    a = std::fmod(a, period);
    return a < 0 ? a + period : a;
  };
  const auto distinct = [&](const Candidate& a, const Candidate& b) {
    if (std::hypot(a.x - b.x, a.y - b.y) > .002F) return true;
    const auto pa = fitter.project(a, true), pb = fitter.project(b, true);
    if (pa.size() != pb.size() || pa.empty()) return true;
    // 以实际围取多边形判断对称等价，而不是只比较角度数值。
    for (const auto& p : pa) {
      float closest = std::numeric_limits<float>::infinity();
      for (const auto& q : pb) closest = std::min(closest, std::hypot(p.x - q.x, p.y - q.y));
      if (closest > .002F) return true;
    }
    for (const auto& p : pb) {
      float closest = std::numeric_limits<float>::infinity();
      for (const auto& q : pa) closest = std::min(closest, std::hypot(p.x - q.x, p.y - q.y));
      if (closest > .002F) return true;
    }
    return false;
  };
  // 模型接受须同时通过框、边缘、可见比例和重叠检查；低重投影误差不能证明外参本身正确。
  const auto valid = [&](const Candidate& c) {
    // 只有已识别前景遮挡且当前候选确实有被遮边缘时，允许较低的可见区域重叠。
    // 未解释的缺边/异类混入不能借此放行；类别纯净及解锁仍由围取许可独立检查。
    const bool explained_occlusion = !options.occluders.empty() && c.visible < .95F;
    const float minimum_overlap = explained_occlusion ? .60F : .72F;
    const float edge_limit = d.class_id == 2 ? std::max(tolerance, 7.F) : tolerance;
    const float overlap_limit = d.class_id == 2 ? .60F : minimum_overlap;
    return c.bbox_error <= tolerance && c.edge_error <= edge_limit && c.visible >= .55F &&
           fitter.overlap(c) >= overlap_limit;
  };
  const auto finish = [&](std::vector<Candidate> candidates) {
    result.evaluations = fitter.evaluations;
    if (fitter.budget()) return budgetResult();
    std::sort(candidates.begin(), candidates.end(),
              [](Candidate a, Candidate b) { return a.cost < b.cost; });
    result.reason = uint8_t(ModelFitReason::EVIDENCE_MISMATCH);
    std::vector<Candidate> retained;
    for (auto candidate : candidates) {
      // 先排除必然因代价或重复而丢弃的候选，再算昂贵的当前像素IoU。
      // 比较的始终是已接受候选；不改变排序、门限、多解或最终保留集合。
      if (!retained.empty() && candidate.cost > retained.front().cost + .50F) continue;
      bool duplicate = false;
      for (const auto& prior : retained)
        if (!distinct(candidate, prior)) duplicate = true;
      if (duplicate || !valid(candidate)) continue;
      retained.push_back(candidate);
      if (retained.size() == 6) break;
    }
    if (fitter.budget()) return budgetResult();
    if (retained.empty()) return result;
    const auto best = retained.front();
    result.x = best.x;
    result.y = best.y;
    result.yaw = normalize(best.yaw);
    result.error_px = best.bbox_error;
    result.visible_fraction = best.visible;
    std::vector<Point2D> envelope;
    for (const auto& c : retained) {
      auto polygon = fitter.project(c, true);
      result.candidates.push_back({c.x, c.y, normalize(c.yaw), c.bbox_error,
                                   std::vector<Point2D>(polygon.begin(), polygon.end())});
      envelope.insert(envelope.end(), polygon.begin(), polygon.end());
    }
    result.footprint = convexHull(std::move(envelope));
    result.status = retained.size() > 1 ? ModelFitStatus::ENVELOPE : ModelFitStatus::FITTED;
    result.reason = uint8_t(ModelFitReason::ACCEPTED);
    return result;
  };
  // 历史姿态只能当初值；必须在当前图像完成细化，并通过与全量拟合相同的证据检查后才返回。
  if (!options.force_full && !options.seeds.empty()) {
    result.search_mode = 2;
    std::vector<Candidate> local;
    for (size_t i = 0; i < std::min<size_t>(6, options.seeds.size()); ++i) {
      const auto& seed = options.seeds[i];
      if (!std::isfinite(seed.x) || !std::isfinite(seed.y) || !std::isfinite(seed.yaw)) continue;
      Candidate c;
      c.x = seed.x;
      c.y = seed.y;
      c.yaw = normalize(seed.yaw);
      local.push_back(fitter.refine(c, true));
      if (fitter.budget()) {
        result.evaluations = fitter.evaluations;
        return budgetResult();
      }
    }
    auto current = finish(local);
    if (current.status == ModelFitStatus::BUDGET) return current;
    if (current.usable()) {
      std::array<Candidate, 6> accepted{};
      size_t accepted_count = 0;
      for (const auto& mode : current.candidates) {
        Candidate c;
        c.x = mode.x;
        c.y = mode.y;
        c.yaw = mode.yaw;
        accepted[accepted_count++] = fitter.evaluate(c, true);
      }
      const auto best = accepted[0];
      bool alternative = false;
      // 六个低成本方向探测检查种子未覆盖的其他可信分支；它们不是完整误差界，仍须周期性全量搜索。
      for (int i = 0; i < 6; ++i) {
        Candidate probe;
        probe.x = coarse.pose_x;
        probe.y = coarse.pose_y;
        probe.yaw = period * i / 6;
        // 蓝色沿用原搜索：配对测试未见初始化收益，且危险物不降低最终模型要求。
        if(d.class_id==3 || !fitter.initializeFromBox(&probe)) probe = fitter.refine(probe, false);
        probe = fitter.evaluate(probe, true);
        bool covered = false;
        for (size_t j = 0; j < accepted_count; ++j)
          if (!distinct(accepted[j], probe)) covered = true;
        if (!covered && probe.cost <= best.cost + .50F && valid(probe)) alternative = true;
        if (fitter.budget()) {
          result.evaluations = fitter.evaluations;
          return budgetResult();
        }
      }
      if (!alternative) {
        current.evaluations = fitter.evaluations;
        return current;
      }
    }
    result = ModelFitResult{};
    result.status = ModelFitStatus::REJECTED;
    result.local_fallbacks = 1;
  }
  result.search_mode = 1;
  std::vector<Candidate> candidates;
  // 成功返回前完成对称周期内的全局朝向覆盖；分批检查时钟，并在假设间检查预算。即使无时间截止，也保
  // 留6000次评价硬上限。
  // 绿/黑允许约15度朝向误差：降低全局初扫密度，随后仍局部细化并走相同本帧接受条件。
  // 橙色长方体和危险蓝色保留5度初扫，避免用它们的精度换算力。
  const float yaw_step=(d.class_id==0 || d.class_id==1)?pi/12:pi/36;
  const int count = static_cast<int>(std::ceil(period / yaw_step));
  for (int i = 0; i < count; ++i) {
    Candidate c;
    c.x = coarse.pose_x;
    c.y = coarse.pose_y;
    c.yaw = period * i / count;
    if(d.class_id==3 || !fitter.initializeFromBox(&c)) c = fitter.refine(c, false);
    c = fitter.evaluate(c, true);
    candidates.push_back(c);
    if (fitter.budget()) {
      result.evaluations = fitter.evaluations;
      return budgetResult();
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](Candidate a, Candidate b) { return a.cost < b.cost; });
  // 先保留互相分离的搜索分支，再做精细化；不能让相邻朝向占满名额，遗漏不同的可信姿态。
  std::vector<Candidate> diverse;
  for (auto c : candidates) {
    bool duplicate = false;
    for (const auto& prior : diverse)
      if (!distinct(c, prior)) duplicate = true;
    if (!duplicate) diverse.push_back(c);
    if (diverse.size() == 12) break;
  }
  for (auto& c : diverse) {
    c = fitter.refine(c, true);
    if (fitter.budget()) {
      result.evaluations = fitter.evaluations;
      return budgetResult();
    }
  }
  return finish(diverse);
}
}  // 命名空间结束 rescue_vision

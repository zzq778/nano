// 模块职责：已知比赛尺寸的地面模型拟合：比较当前颜色轮廓和投影，保留可信多解，失败不假装成功。
#pragma once

#include <rescue_vision/ground_geometry.hpp>
#include <rescue_vision/rgb_frame_view.hpp>
#include <chrono>

namespace rescue_vision {
enum class ModelFitStatus : uint8_t {
  UNAVAILABLE = 0,
  FITTED = 1,
  ENVELOPE = 2,
  REJECTED = 3,
  GROUP = 4,
  BUDGET = 5
};
// 诊断原因与可用状态分开；只有FITTED和ENVELOPE表示模型可用于围取许可评价。
enum class ModelFitReason : uint8_t {
  NONE = 0,
  INVALID_INPUT = 1,
  INSUFFICIENT_EVIDENCE = 2,
  EVIDENCE_MISMATCH = 3,
  GROUP = 4,
  BUDGET = 5,
  ACCEPTED = 6
};
struct ModelPoseCandidate {
  float x{0}, y{0}, yaw{0}, error_px{-1};
  std::vector<Point2D> footprint;
};
struct ModelFitScratch {
  std::vector<uint8_t> mask, hidden;
  std::vector<float> distance;
  std::vector<int> labels, queue;
};
struct ModelFitOptions {
  // 上一帧候选只是初值，每个返回结果仍须由当前RGB重新检验。
  std::vector<ModelPoseCandidate> seeds;
  // 只接收调用者已验证近远次序的前景框；不明原因的缺边不能自动解释为遮挡。
  std::vector<BoundingBox2D> occluders;
  std::vector<uint8_t> occluder_classes;  // 与遮挡框配对，只用当前颜色像素而非整框内部
  std::chrono::steady_clock::time_point deadline{std::chrono::steady_clock::time_point::max()};
  bool force_full{false};
  ModelFitScratch* scratch{nullptr};  // 借用scratch，同一时刻只能供一次拟合使用
};
struct ModelFitResult {
  ModelFitStatus status{ModelFitStatus::UNAVAILABLE};
  float x{0}, y{0}, yaw{0}, error_px{-1};
  unsigned evaluations{0};
  unsigned local_fallbacks{0};
  std::vector<ModelPoseCandidate> candidates;  // 最多六个彼此分离的地面姿态候选
  uint8_t search_mode{0};                      // 0未搜索，1全量搜索，2局部细化
  uint8_t reason{uint8_t(ModelFitReason::INVALID_INPUT)};
  float visible_fraction{0};
  std::vector<Point2D> footprint;
  // Diagnostic optimum, retained even if acceptance/budget fails. Not a fitted pose.
  ModelPoseCandidate best_effort;
  bool best_effort_available{false};
  float best_effort_cost{1e9F};
  bool usable() const {
    return status == ModelFitStatus::FITTED || status == ModelFitStatus::ENVELOPE;
  }
};
// 以比赛尺寸和地面支撑约束，把模型投影与框及当前颜色轮廓对齐；粗观察仅作初值，不是拟合失败后的成功
// 回退。
ModelFitResult fitGroundModel(const Detection2D& detection, const RgbFrameView& rgb,
                              const CameraIntrinsics& camera,
                              const RigidTransform3D& camera_to_base,
                              const GeometricObservation& coarse,
                              const ModelFitOptions& options = {});
}  // 命名空间结束 rescue_vision

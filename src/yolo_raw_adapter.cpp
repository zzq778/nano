// 模块职责：YOLO原始输出布局与类别顺序适配；统一为候选框后再交给后处理，不在这里做跟踪。
#include <rescue_vision/yolo_raw_adapter.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace rescue_vision {
namespace {

bool isSingleton(int64_t value) { return value == 1; }

std::vector<int64_t> squeeze(const std::vector<int64_t>& shape) {
  std::vector<int64_t> result;
  for (const int64_t dimension : shape) {
    if (!isSingleton(dimension)) result.push_back(dimension);
  }
  return result;
}

bool featureLayout(int64_t features, bool* has_objectness) {
  if (features == static_cast<int64_t>(5 + YoloRawAdapter::kClassCount)) {
    *has_objectness = true;
    return true;
  }
  if (features == static_cast<int64_t>(4 + YoloRawAdapter::kClassCount)) {
    *has_objectness = false;
    return true;
  }
  return false;
}

float valueAt(const float* tensor, const YoloRawSpec& spec, std::size_t candidate,
              std::size_t feature) {
  const bool rows = spec.layout == YoloRawLayout::ROWS_WITH_OBJECTNESS ||
                    spec.layout == YoloRawLayout::ROWS_WITHOUT_OBJECTNESS;
  return rows ? tensor[candidate * spec.feature_count + feature]
              : tensor[feature * spec.candidate_count + candidate];
}

// 部署模型训练通道为橙、蓝、黑、绿、物资区、伤员区；语义type为NORMAL、CORE、CASUALTY、DANGER、物资区、伤员区。必须先映射，
// 再选择阈值或发布，不能直接把模型下标当类别ID。
constexpr std::array<std::size_t, YoloRawAdapter::kClassCount> kModelToSemanticClass{{2, 3, 1, 0, 4, 5}};

}  // 命名空间结束

bool YoloRawAdapter::inspect(const std::vector<int64_t>& shape, YoloRawSpec* spec,
                             std::string* error) {
  if (spec == nullptr) {
    if (error) *error = "output spec pointer is null";
    return false;
  }
  for (const int64_t dimension : shape) {
    if (dimension <= 0) {
      if (error) *error = "dynamic or empty output dimensions are unsupported";
      return false;
    }
  }
  const std::vector<int64_t> dimensions = squeeze(shape);
  if (dimensions.size() != 2) {
    if (error) *error = "raw YOLO output must reduce to a two-dimensional tensor";
    return false;
  }

  bool rows_objectness = false;
  bool channels_objectness = false;
  const bool rows_valid = featureLayout(dimensions[1], &rows_objectness);
  const bool channels_valid = featureLayout(dimensions[0], &channels_objectness);
  if (rows_valid == channels_valid) {
    if (error) {
      *error = rows_valid ? "raw YOLO output layout is ambiguous"
                          : "raw YOLO output feature count is neither 10 nor 11 for six classes";
    }
    return false;
  }

  if (rows_valid) {
    spec->candidate_count = static_cast<std::size_t>(dimensions[0]);
    spec->feature_count = static_cast<std::size_t>(dimensions[1]);
    spec->layout = rows_objectness ? YoloRawLayout::ROWS_WITH_OBJECTNESS
                                   : YoloRawLayout::ROWS_WITHOUT_OBJECTNESS;
  } else {
    spec->candidate_count = static_cast<std::size_t>(dimensions[1]);
    spec->feature_count = static_cast<std::size_t>(dimensions[0]);
    spec->layout = channels_objectness ? YoloRawLayout::CHANNELS_WITH_OBJECTNESS
                                       : YoloRawLayout::CHANNELS_WITHOUT_OBJECTNESS;
  }
  if (spec->candidate_count == 0) {
    if (error) *error = "raw YOLO output contains no candidates";
    return false;
  }
  return true;
}

bool YoloRawAdapter::adapt(const float* tensor, std::size_t element_count,
                           const std::vector<int64_t>& shape,
                           const std::array<float, kClassCount>& thresholds,
                           std::vector<RawDetection>* detections, std::string* error) {
  if (tensor == nullptr || detections == nullptr) {
    if (error) *error = "raw tensor or destination pointer is null";
    return false;
  }
  for (const float threshold : thresholds) {
    if (!std::isfinite(threshold) || threshold < 0.0F || threshold > 1.0F) {
      if (error) *error = "class confidence threshold is outside [0,1]";
      return false;
    }
  }

  YoloRawSpec spec;
  if (!inspect(shape, &spec, error)) return false;
  if (spec.candidate_count > std::numeric_limits<std::size_t>::max() / spec.feature_count ||
      spec.candidate_count * spec.feature_count != element_count) {
    if (error) *error = "raw tensor element count does not match its shape";
    return false;
  }

  const bool has_objectness = spec.layout == YoloRawLayout::ROWS_WITH_OBJECTNESS ||
                              spec.layout == YoloRawLayout::CHANNELS_WITH_OBJECTNESS;
  const std::size_t class_offset = has_objectness ? 5U : 4U;
  detections->clear();
  detections->reserve(spec.candidate_count);
  for (std::size_t candidate = 0; candidate < spec.candidate_count; ++candidate) {
    const float cx = valueAt(tensor, spec, candidate, 0);
    const float cy = valueAt(tensor, spec, candidate, 1);
    const float width = valueAt(tensor, spec, candidate, 2);
    const float height = valueAt(tensor, spec, candidate, 3);
    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(width) ||
        !std::isfinite(height) || width <= 0.0F || height <= 0.0F) {
      continue;
    }

    const float objectness = has_objectness ? valueAt(tensor, spec, candidate, 4) : 1.0F;
    if (!std::isfinite(objectness) || objectness < 0.0F) continue;
    std::size_t best_class = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (std::size_t class_id = 0; class_id < kClassCount; ++class_id) {
      const float score = valueAt(tensor, spec, candidate, class_offset + class_id);
      if (std::isfinite(score) && score > best_score) {
        best_score = score;
        best_class = class_id;
      }
    }
    const float confidence = objectness * best_score;
    const std::size_t semantic_class = kModelToSemanticClass[best_class];
    if (!std::isfinite(confidence) || confidence < thresholds[semantic_class]) continue;

    RawDetection detection;
    detection.class_id = static_cast<uint8_t>(semantic_class);
    detection.confidence = confidence;
    detection.center_x = cx;
    detection.center_y = cy;
    detection.width = width;
    detection.height = height;
    detections->push_back(detection);
  }
  return true;
}

}  // 命名空间结束 rescue_vision

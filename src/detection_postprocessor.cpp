// 模块职责：恢复letterbox前的像素框并做按类NMS；输出仍是外接框，不是真实轮廓。
#include <rescue_vision/detection_postprocessor.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace rescue_vision {

bool makeLetterboxTransform(int source_width, int source_height, int input_width, int input_height,
                            LetterboxTransform* transform) {
  if (transform == nullptr || source_width <= 0 || source_height <= 0 || input_width <= 0 ||
      input_height <= 0) {
    return false;
  }
  const float scale = std::min(static_cast<float>(input_width) / source_width,
                               static_cast<float>(input_height) / source_height);
  const int resized_width = std::max(1, static_cast<int>(std::round(source_width * scale)));
  const int resized_height = std::max(1, static_cast<int>(std::round(source_height * scale)));
  transform->scale = scale;
  // 奇数填充拆分时，与OpenCV实际使用的整数上/左边距一致，避免框反变换偏移。
  transform->pad_x = static_cast<float>((input_width - resized_width) / 2);
  transform->pad_y = static_cast<float>((input_height - resized_height) / 2);
  transform->source_width = source_width;
  transform->source_height = source_height;
  transform->input_width = input_width;
  transform->input_height = input_height;
  transform->resized_width = resized_width;
  transform->resized_height = resized_height;
  return true;
}

bool restoreDetection(const RawDetection& raw, const LetterboxTransform& transform,
                      Detection2D* detection) {
  if (detection == nullptr || transform.scale <= 0.0F || transform.source_width <= 0 ||
      transform.source_height <= 0) {
    return false;
  }
  const float network_x0 = raw.center_x - 0.5F * raw.width;
  const float network_y0 = raw.center_y - 0.5F * raw.height;
  const float network_x1 = raw.center_x + 0.5F * raw.width;
  const float network_y1 = raw.center_y + 0.5F * raw.height;
  BoundingBox2D box;
  box.x_min = (network_x0 - transform.pad_x) / transform.scale;
  box.y_min = (network_y0 - transform.pad_y) / transform.scale;
  box.x_max = (network_x1 - transform.pad_x) / transform.scale;
  box.y_max = (network_y1 - transform.pad_y) / transform.scale;
  box.x_min = std::max(0.0F, std::min(box.x_min, static_cast<float>(transform.source_width)));
  box.y_min = std::max(0.0F, std::min(box.y_min, static_cast<float>(transform.source_height)));
  box.x_max = std::max(0.0F, std::min(box.x_max, static_cast<float>(transform.source_width)));
  box.y_max = std::max(0.0F, std::min(box.y_max, static_cast<float>(transform.source_height)));
  if (!(box.x_max > box.x_min && box.y_max > box.y_min)) return false;
  detection->class_id = raw.class_id;
  detection->confidence = raw.confidence;
  detection->bbox = box;
  return true;
}

float intersectionOverUnion(const BoundingBox2D& left, const BoundingBox2D& right) {
  const float x0 = std::max(left.x_min, right.x_min);
  const float y0 = std::max(left.y_min, right.y_min);
  const float x1 = std::min(left.x_max, right.x_max);
  const float y1 = std::min(left.y_max, right.y_max);
  const float intersection = std::max(0.0F, x1 - x0) * std::max(0.0F, y1 - y0);
  const float left_area =
      std::max(0.0F, left.x_max - left.x_min) * std::max(0.0F, left.y_max - left.y_min);
  const float right_area =
      std::max(0.0F, right.x_max - right.x_min) * std::max(0.0F, right.y_max - right.y_min);
  const float union_area = left_area + right_area - intersection;
  return union_area > 0.0F ? intersection / union_area : 0.0F;
}

std::vector<Detection2D> classWiseNms(const std::vector<Detection2D>& detections,
                                      float iou_threshold) {
  std::vector<std::size_t> order(detections.size());
  for (std::size_t index = 0; index < order.size(); ++index) order[index] = index;
  std::stable_sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
    return detections[left].confidence > detections[right].confidence;
  });

  std::vector<Detection2D> kept;
  // Model class_id is a byte. Compare a candidate only against accepted boxes
  // of that class, while appending to kept in the same global stable-score
  // order as before. This is exactly the original class-wise suppression rule.
  std::array<std::vector<std::size_t>, 256> accepted_by_class;
  for (const std::size_t index : order) {
    const Detection2D& candidate = detections[index];
    auto& same_class = accepted_by_class[candidate.class_id];
    bool suppressed = false;
    for (const std::size_t accepted_index : same_class) {
      if (intersectionOverUnion(candidate.bbox, kept[accepted_index].bbox) > iou_threshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      same_class.push_back(kept.size());
      kept.push_back(candidate);
    }
  }
  return kept;
}

}  // 命名空间结束 rescue_vision

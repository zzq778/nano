// 模块职责：YOLO原始输出布局与类别顺序适配；统一为候选框后再交给后处理，不在这里做跟踪。
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rescue_vision {

struct RawDetection {
  uint8_t class_id{0};
  float confidence{0.0F};
  float center_x{0.0F};
  float center_y{0.0F};
  float width{0.0F};
  float height{0.0F};
};

enum class YoloRawLayout {
  ROWS_WITH_OBJECTNESS,
  ROWS_WITHOUT_OBJECTNESS,
  CHANNELS_WITH_OBJECTNESS,
  CHANNELS_WITHOUT_OBJECTNESS,
};

struct YoloRawSpec {
  YoloRawLayout layout{YoloRawLayout::ROWS_WITH_OBJECTNESS};
  std::size_t candidate_count{0};
  std::size_t feature_count{0};
};

// 把四类YOLO原始张量转换成与模型布局无关的候选列表；NMS、letterbox反变换、深度几何和跟踪由后续模块
// 负责。
class YoloRawAdapter {
 public:
  static constexpr std::size_t kClassCount = 6;

  static bool inspect(const std::vector<int64_t>& shape, YoloRawSpec* spec, std::string* error);

  static bool adapt(const float* tensor, std::size_t element_count,
                    const std::vector<int64_t>& shape,
                    const std::array<float, kClassCount>& thresholds,
                    std::vector<RawDetection>* detections, std::string* error);
};

}  // 命名空间结束 rescue_vision

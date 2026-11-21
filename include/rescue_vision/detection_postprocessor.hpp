// 模块职责：恢复letterbox前的像素框并做按类NMS；输出仍是外接框，不是真实轮廓。
#pragma once

#include <vector>

#include <rescue_vision/yolo_raw_adapter.hpp>

namespace rescue_vision {

struct BoundingBox2D {
  float x_min{0.0F};
  float y_min{0.0F};
  float x_max{0.0F};
  float y_max{0.0F};
};

struct Detection2D {
  uint8_t class_id{0};
  float confidence{0.0F};
  BoundingBox2D bbox;
};

struct LetterboxTransform {
  float scale{1.0F};
  float pad_x{0.0F};
  float pad_y{0.0F};
  int source_width{0};
  int source_height{0};
  int input_width{0};
  int input_height{0};
  int resized_width{0};
  int resized_height{0};
};

bool makeLetterboxTransform(int source_width, int source_height, int input_width, int input_height,
                            LetterboxTransform* transform);

bool restoreDetection(const RawDetection& raw, const LetterboxTransform& transform,
                      Detection2D* detection);

float intersectionOverUnion(const BoundingBox2D& left, const BoundingBox2D& right);

// NMS按类别分别执行；不同语义类别的重叠框不能互相压掉。
std::vector<Detection2D> classWiseNms(const std::vector<Detection2D>& detections,
                                      float iou_threshold);

}  // 命名空间结束 rescue_vision

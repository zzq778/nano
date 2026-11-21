// 模块职责：TensorRT引擎与CUDA资源拥有者：加载绑定、复用设备缓冲并串行执行推理。
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rescue_vision {

struct TensorBindingInfo {
  std::string name;
  std::vector<int64_t> dimensions;
  std::size_t element_count{0};
  bool input{false};
  bool fp16{false};
};

struct TensorRtEngineInfo {
  int input_width{0};
  int input_height{0};
  TensorBindingInfo input;
  TensorBindingInfo output;
};

class TensorRtEngine {
 public:
  TensorRtEngine();
  ~TensorRtEngine();
  TensorRtEngine(TensorRtEngine&&) noexcept;
  TensorRtEngine& operator=(TensorRtEngine&&) noexcept;
  TensorRtEngine(const TensorRtEngine&) = delete;
  TensorRtEngine& operator=(const TensorRtEngine&) = delete;

  bool load(const std::string& path, std::string* error);
  void reset();
  bool ready() const;
  const TensorRtEngineInfo& info() const;

  // 输入必须已经letterbox、归一化并转成浮点NCHW；本方法只拥有TensorRT执行与数据传输，图像预处理在外
  // 层。
  bool infer(const float* input, std::size_t element_count, std::vector<float>* output,
             std::string* error);

  static bool compiledWithTensorRt();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // 命名空间结束 rescue_vision

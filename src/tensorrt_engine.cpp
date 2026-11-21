// 模块职责：TensorRT引擎与CUDA资源拥有者：加载绑定、复用设备缓冲并串行执行推理。
#include <rescue_vision/tensorrt_engine.hpp>
#include <rescue_vision/yolo_raw_adapter.hpp>

#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

#ifdef RESCUE_VISION_HAS_TENSORRT
#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#endif

namespace rescue_vision {

class TensorRtEngine::Impl {
 public:
  TensorRtEngineInfo info;

#ifdef RESCUE_VISION_HAS_TENSORRT
  class Logger : public nvinfer1::ILogger {
   public:
    void log(Severity severity, const char* message) noexcept override {
      if (severity <= Severity::kWARNING) last_message = message ? message : "";
    }
    std::string last_message;
  } logger;

  template <typename T>
  struct TrtDestroy {
    void operator()(T* object) const {
      if (object != nullptr) object->destroy();
    }
  };

  std::unique_ptr<nvinfer1::IRuntime, TrtDestroy<nvinfer1::IRuntime>> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDestroy<nvinfer1::ICudaEngine>> engine;
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDestroy<nvinfer1::IExecutionContext>> context;
  std::vector<void*> device_bindings;
  // infer由单一处理回调串行调用；跨帧复用转换存储，更换引擎时清理，不能多线程共享此可写状态。
  std::vector<__half> half_input;
  std::vector<__half> half_output;
  int input_index{-1};
  int output_index{-1};

  ~Impl() { clear(); }

  void clear() {
    for (void* pointer : device_bindings) {
      if (pointer != nullptr) cudaFree(pointer);
    }
    device_bindings.clear();
    half_input.clear();
    half_output.clear();
    context.reset();
    engine.reset();
    runtime.reset();
    input_index = -1;
    output_index = -1;
    info = TensorRtEngineInfo{};
  }
#else
  void clear() { info = TensorRtEngineInfo{}; }
#endif
};

namespace {

#ifdef RESCUE_VISION_HAS_TENSORRT
bool readFile(const std::string& path, std::vector<char>* bytes, std::string* error) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    if (error) *error = "cannot open TensorRT engine: " + path;
    return false;
  }
  const std::streamoff size = stream.tellg();
  if (size <= 0 || static_cast<unsigned long long>(size) >
                       static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
    if (error) *error = "TensorRT engine file is empty or too large";
    return false;
  }
  bytes->resize(static_cast<std::size_t>(size));
  stream.seekg(0, std::ios::beg);
  if (!stream.read(bytes->data(), size)) {
    if (error) *error = "failed to read complete TensorRT engine file";
    return false;
  }
  return true;
}

bool tensorInfo(const nvinfer1::ICudaEngine& engine, int binding, TensorBindingInfo* info,
                std::string* error) {
  // 适配器按紧排列张量复制CUDA内存；向量化绑定可能要求补齐通道，不能当作普通逻辑形状直接复制。
  if (engine.getBindingFormat(binding) != nvinfer1::TensorFormat::kLINEAR ||
      engine.getLocation(binding) != nvinfer1::TensorLocation::kDEVICE) {
    if (error) *error = "TensorRT binding must have LINEAR device storage";
    return false;
  }
  const nvinfer1::Dims dimensions = engine.getBindingDimensions(binding);
  if (dimensions.nbDims <= 0) {
    if (error) *error = "TensorRT binding has no dimensions";
    return false;
  }
  std::size_t elements = 1;
  info->dimensions.clear();
  for (int i = 0; i < dimensions.nbDims; ++i) {
    if (dimensions.d[i] <= 0) {
      if (error) *error = "dynamic TensorRT binding dimensions are unsupported";
      return false;
    }
    const std::size_t dimension = static_cast<std::size_t>(dimensions.d[i]);
    if (elements > std::numeric_limits<std::size_t>::max() / dimension) {
      if (error) *error = "TensorRT binding element count overflow";
      return false;
    }
    elements *= dimension;
    info->dimensions.push_back(dimensions.d[i]);
  }
  const nvinfer1::DataType type = engine.getBindingDataType(binding);
  if (type != nvinfer1::DataType::kFLOAT && type != nvinfer1::DataType::kHALF) {
    if (error) *error = "only FP32 and FP16 TensorRT bindings are supported";
    return false;
  }
  const std::size_t bytes_per_element =
      type == nvinfer1::DataType::kHALF ? sizeof(__half) : sizeof(float);
  if (elements > std::numeric_limits<std::size_t>::max() / bytes_per_element) {
    if (error) *error = "TensorRT binding storage size overflow";
    return false;
  }
  info->name = engine.getBindingName(binding) ? engine.getBindingName(binding) : "";
  info->element_count = elements;
  info->input = engine.bindingIsInput(binding);
  info->fp16 = type == nvinfer1::DataType::kHALF;
  return true;
}

std::size_t storageBytes(const TensorBindingInfo& info) {
  return info.element_count * (info.fp16 ? sizeof(__half) : sizeof(float));
}

bool cudaSucceeded(cudaError_t code, const char* operation, std::string* error) {
  if (code == cudaSuccess) return true;
  if (error) {
    *error = std::string(operation) + ": " + cudaGetErrorString(code);
  }
  return false;
}
#endif

}  // 命名空间结束

TensorRtEngine::TensorRtEngine() : impl_(new Impl) {}
TensorRtEngine::~TensorRtEngine() = default;
TensorRtEngine::TensorRtEngine(TensorRtEngine&&) noexcept = default;
TensorRtEngine& TensorRtEngine::operator=(TensorRtEngine&&) noexcept = default;

bool TensorRtEngine::compiledWithTensorRt() {
#ifdef RESCUE_VISION_HAS_TENSORRT
  return true;
#else
  return false;
#endif
}

bool TensorRtEngine::load(const std::string& path, std::string* error) {
  reset();
#ifndef RESCUE_VISION_HAS_TENSORRT
  (void)path;
  if (error) *error = "rescue_vision was built without TensorRT and CUDA support";
  return false;
#else
  std::vector<char> bytes;
  if (!readFile(path, &bytes, error)) return false;

  impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
  if (!impl_->runtime) {
    if (error) *error = "createInferRuntime failed: " + impl_->logger.last_message;
    reset();
    return false;
  }
  impl_->engine.reset(impl_->runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
  if (!impl_->engine) {
    if (error) *error = "deserializeCudaEngine failed: " + impl_->logger.last_message;
    reset();
    return false;
  }
  if (impl_->engine->hasImplicitBatchDimension()) {
    if (error) *error = "implicit-batch TensorRT engines are unsupported; export explicit batch=1";
    reset();
    return false;
  }

  const int binding_count = impl_->engine->getNbBindings();
  if (binding_count != 2) {
    if (error) *error = "engine must expose exactly one input and one raw output binding";
    reset();
    return false;
  }
  for (int binding = 0; binding < binding_count; ++binding) {
    TensorBindingInfo info;
    if (!tensorInfo(*impl_->engine, binding, &info, error)) {
      reset();
      return false;
    }
    if (info.input) {
      if (impl_->input_index >= 0) {
        if (error) *error = "engine exposes more than one input binding";
        reset();
        return false;
      }
      impl_->input_index = binding;
      impl_->info.input = info;
    } else {
      if (impl_->output_index >= 0) {
        if (error) *error = "engine exposes more than one output binding";
        reset();
        return false;
      }
      impl_->output_index = binding;
      impl_->info.output = info;
    }
  }
  if (impl_->input_index < 0 || impl_->output_index < 0 ||
      impl_->info.input.dimensions.size() != 4 || impl_->info.input.dimensions[0] != 1 ||
      impl_->info.input.dimensions[1] != 3) {
    if (error) *error = "engine input must be fixed NCHW [1,3,H,W]";
    reset();
    return false;
  }
  impl_->info.input_height = static_cast<int>(impl_->info.input.dimensions[2]);
  impl_->info.input_width = static_cast<int>(impl_->info.input.dimensions[3]);

  YoloRawSpec output_spec;
  if (!YoloRawAdapter::inspect(impl_->info.output.dimensions, &output_spec, error)) {
    if (error) *error = "unsupported TensorRT raw output: " + *error;
    reset();
    return false;
  }

  impl_->context.reset(impl_->engine->createExecutionContext());
  if (!impl_->context) {
    if (error) *error = "createExecutionContext failed";
    reset();
    return false;
  }
  impl_->device_bindings.assign(static_cast<std::size_t>(binding_count), nullptr);
  for (int binding = 0; binding < binding_count; ++binding) {
    const TensorBindingInfo& info =
        binding == impl_->input_index ? impl_->info.input : impl_->info.output;
    if (!cudaSucceeded(cudaMalloc(&impl_->device_bindings[binding], storageBytes(info)),
                       "cudaMalloc TensorRT binding", error)) {
      reset();
      return false;
    }
  }
  // 在发布“模型就绪”和订阅图像之前完成CUDA首次执行初始化。
  // 这些零张量只预热执行资源，输出全部丢弃，不产生任何物块观察。
  std::vector<float> warm_input(impl_->info.input.element_count,0.0F), warm_output;
  for (unsigned pass=0;pass<2;++pass) {
    if (!infer(warm_input.data(),warm_input.size(),&warm_output,error)) {
      reset();
      return false;
    }
  }
  return true;
#endif
}

void TensorRtEngine::reset() { impl_->clear(); }

bool TensorRtEngine::ready() const {
#ifdef RESCUE_VISION_HAS_TENSORRT
  return impl_->engine != nullptr && impl_->context != nullptr;
#else
  return false;
#endif
}

const TensorRtEngineInfo& TensorRtEngine::info() const { return impl_->info; }

bool TensorRtEngine::infer(const float* input, std::size_t element_count,
                           std::vector<float>* output, std::string* error) {
#ifndef RESCUE_VISION_HAS_TENSORRT
  (void)input;
  (void)element_count;
  (void)output;
  if (error) *error = "rescue_vision was built without TensorRT and CUDA support";
  return false;
#else
  if (!ready() || input == nullptr || output == nullptr) {
    if (error) *error = "TensorRT engine is not ready or infer arguments are null";
    return false;
  }
  if (element_count != impl_->info.input.element_count) {
    if (error) *error = "preprocessed input element count does not match engine binding";
    return false;
  }

  const void* host_input = input;
  if (impl_->info.input.fp16) {
    impl_->half_input.resize(element_count);
    for (std::size_t i = 0; i < element_count; ++i) impl_->half_input[i] = __float2half(input[i]);
    host_input = impl_->half_input.data();
  }
  if (!cudaSucceeded(cudaMemcpy(impl_->device_bindings[impl_->input_index], host_input,
                                storageBytes(impl_->info.input), cudaMemcpyHostToDevice),
                     "cudaMemcpy input", error))
    return false;
  if (!impl_->context->executeV2(impl_->device_bindings.data())) {
    if (error) *error = "TensorRT executeV2 failed";
    return false;
  }

  output->resize(impl_->info.output.element_count);
  if (impl_->info.output.fp16) {
    impl_->half_output.resize(impl_->info.output.element_count);
    if (!cudaSucceeded(
            cudaMemcpy(impl_->half_output.data(), impl_->device_bindings[impl_->output_index],
                       storageBytes(impl_->info.output), cudaMemcpyDeviceToHost),
            "cudaMemcpy FP16 output", error))
      return false;
    for (std::size_t i = 0; i < impl_->half_output.size(); ++i) {
      (*output)[i] = __half2float(impl_->half_output[i]);
    }
  } else if (!cudaSucceeded(cudaMemcpy(output->data(), impl_->device_bindings[impl_->output_index],
                                       storageBytes(impl_->info.output), cudaMemcpyDeviceToHost),
                            "cudaMemcpy FP32 output", error)) {
    return false;
  }
  return true;
#endif
}

}  // 命名空间结束 rescue_vision

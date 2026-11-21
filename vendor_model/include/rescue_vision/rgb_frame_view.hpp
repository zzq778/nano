// 模块职责：借用RGB8图像视图：像素指针与行跨度分离，存储生命周期由上层消息拥有者保证。
#pragma once
#include <cstdint>
namespace rescue_vision {
// 借用RGB8字节；生产者负责保证存储在整次处理调用中存活。
struct RgbFrameView {
  const uint8_t* pixels{nullptr};
  int width{0}, height{0}, stride{0};  // 行跨度为字节数，允许行填充
};
}  // 命名空间结束 rescue_vision

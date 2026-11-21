// 模块职责：模型拟合的宽颜色证据掩膜；只辅助轮廓检验，类别权威仍是YOLO。
#pragma once
#include <algorithm>
#include <cstdint>
namespace rescue_vision {
inline bool modelColorMatch(const uint8_t* p, uint8_t cls) {
  const int r = p[0], g = p[1], b = p[2];
  // 颜色掩膜只辅助几何，类别仍由YOLO决定；有彩色任务类别会排除中性背景和暗影。
  if (cls == 0) return g > 20 && g > r * 1.18F && g > b * 1.10F;
  if (cls == 1) return std::max(r, std::max(g, b)) < 95;
  if (cls == 2) return r > 25 && r > g * 1.15F && g > b * 1.15F;
  return b > 20 && g > 20 && b > r * 1.20F && g > r * 1.15F && b > g * .80F;
}
}  // 命名空间结束 rescue_vision

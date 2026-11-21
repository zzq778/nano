// 模块职责：框到地面粗位置与RGB地面射线几何；粗位置不等于成功拟合，也不单独授予围取许可。
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <rescue_vision/detection_postprocessor.hpp>

namespace rescue_vision {

struct Point2D {
  float x{0.0F};
  float y{0.0F};
};

// 针孔内参以像素为单位，必须与实际图像分辨率/裁剪口径一致；数值有限不等于物理标定准确。
struct CameraIntrinsics {
  float fx{0.0F};
  float fy{0.0F};
  float cx{0.0F};
  float cy{0.0F};
  int width{0};
  int height{0};
};

// 行优先旋转矩阵与米制平移：目标坐标=R×源坐标+t。物理TF与离地算法变换不能混用高度基准。
struct RigidTransform3D {
  std::array<float, 9> rotation{{1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F}};
  std::array<float, 3> translation{{0.0F, 0.0F, 0.0F}};
};

// 物理TF以轮轴为原点，算法仍使用同一XY轴但高度从地面算起；只在流水线入口归一化一次，其他模型或聚类
// 不能再次叠加地面偏移。
inline bool groundReferencedTransform(const RigidTransform3D& physical, float ground_z,
                                      RigidTransform3D* output) {
  if (!output || !std::isfinite(ground_z)) return false;
  for (float x : physical.rotation)
    if (!std::isfinite(x)) return false;
  for (float x : physical.translation)
    if (!std::isfinite(x)) return false;
  auto adjusted = physical;
  adjusted.translation[2] -= ground_z;
  if (!std::isfinite(adjusted.translation[2]) || adjusted.translation[2] <= 0) return false;
  *output = adjusted;
  return true;
}

struct GroundGeometryConfig {
  float task_bbox_oversized_width_max_m{0.100F};
};

// 本帧观察：pose可先是粗位置，模型通过后才细化。geometry_valid和model_fit_status各有语
// 义，不能互相替代。
struct GeometricObservation {
  uint8_t class_id{0};
  float confidence{0.0F};
  BoundingBox2D bbox;
  bool geometry_valid{false};
  bool rgb_ground_anchor_valid{false};
  Point2D rgb_ground_anchor;  // 框底中心射线的地面投影；不是物块中心，也不保证是真实接地点。
  bool bbox_ground_width_valid{false};
  float bbox_ground_width_m{0.0F};
  bool bbox_oversized{false};
  // 初始化时为RGB落地锚点，模型成功后为精细中心；控制接近应使用上方独立rgb_ground_anchor。
  float pose_x{0.0F};
  float pose_y{0.0F};
  float pose_yaw{0.0F};
  bool cluster_hint{false};
  bool group_geometry_valid{false};  // 当前绿色群可见支持，不等价于可主动围取或单体精模型
  std::vector<std::vector<Point2D>> group_support; // 有界小片的并集；片间空白不连接成整体包络
  uint8_t model_fit_status{0};
  float model_fit_error_px{-1.0F};
  float model_fit_time_ms{0.0F};
  uint32_t model_fit_evaluations{0};
  // 所有当前可信模型候选，供最终围取逐个检查；footprint是候选凸包，仅适合保守障碍范围。
  std::vector<std::vector<Point2D>> model_candidate_footprints;
  uint8_t model_search_mode{0};
  uint8_t model_fit_reason{0};
  float model_visible_fraction{0};
  std::vector<Point2D> footprint;
  uint32_t source_detection_index{UINT32_MAX};  // 仅本帧输入下标，不是跟踪身份
};

bool validateCameraIntrinsics(const CameraIntrinsics& intrinsics);

// 同一地面坐标系中的射线/水平面交点。相机参数在帧入口校验一次，采样循环不重复校验。
inline bool rayToGroundHeight(float u,float v,float z,const CameraIntrinsics& k,
 const RigidTransform3D& tf,Point2D* point) {
 const float optical[3]={(u-k.cx)/k.fx,(v-k.cy)/k.fy,1};float ray[3]={};
 for(int a=0;a<3;++a)for(int b=0;b<3;++b)ray[a]+=tf.rotation[a*3+b]*optical[b];
 if(!point||ray[2]>=-1e-5F)return false;
 const float scale=(z-tf.translation[2])/ray[2];
 *point={tf.translation[0]+scale*ray[0],tf.translation[1]+scale*ray[1]};
 return scale>0&&std::isfinite(point->x)&&std::isfinite(point->y)&&std::hypot(point->x,point->y)<4;
}


GeometricObservation estimateGeometry(const Detection2D& detection, const CameraIntrinsics& intrinsics,
                                      const RigidTransform3D& camera_to_base,
                                      const GroundGeometryConfig& config);

}  // 命名空间结束 rescue_vision

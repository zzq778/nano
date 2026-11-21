// 模块职责：RGB框到地面粗位置；粗位置不等于成功拟合，也不单独授予围取许可。
#include <rescue_vision/ground_geometry.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace rescue_vision {
namespace {

constexpr float kPi = 3.14159265358979323846F;

bool projectGround(float u, float v, const CameraIntrinsics& camera,
                   const RigidTransform3D& transform, Point2D* point) {
  if (point == nullptr || !std::isfinite(camera.fx) || !std::isfinite(camera.fy) ||
      camera.fx <= 0.0F || camera.fy <= 0.0F)
    return false;
  const std::array<float, 3> ray_camera{
      {(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, 1.0F}};
  const auto& r = transform.rotation;
  const std::array<float, 3> ray_base{
      {r[0] * ray_camera[0] + r[1] * ray_camera[1] + r[2] * ray_camera[2],
       r[3] * ray_camera[0] + r[4] * ray_camera[1] + r[5] * ray_camera[2],
       r[6] * ray_camera[0] + r[7] * ray_camera[1] + r[8] * ray_camera[2]}};
  const float origin_z = transform.translation[2];
  if (!std::isfinite(origin_z) || !std::isfinite(ray_base[2]) || std::abs(ray_base[2]) < 1.0e-6F)
    return false;
  const float scale = -origin_z / ray_base[2];
  if (!std::isfinite(scale) || scale <= 0.0F) return false;
  const float x = transform.translation[0] + scale * ray_base[0];
  const float y = transform.translation[1] + scale * ray_base[1];
  if (!std::isfinite(x) || !std::isfinite(y)) return false;
  point->x = x;
  point->y = y;
  return true;
}

std::vector<Point2D> rectangle(float cx, float cy, float length, float width, float yaw) {
  const float c = std::cos(yaw);
  const float s = std::sin(yaw);
  const float half_l = 0.5F * length;
  const float half_w = 0.5F * width;
  const std::array<Point2D, 4> local{
      {{-half_l, -half_w}, {half_l, -half_w}, {half_l, half_w}, {-half_l, half_w}}};
  std::vector<Point2D> result;
  result.reserve(4);
  for (const Point2D& point : local) {
    result.push_back({cx + c * point.x - s * point.y, cy + s * point.x + c * point.y});
  }
  return result;  // 逆时针顶点序列，不重复首点
}

std::vector<Point2D> conservativeCasualtyFootprint(float cx, float cy) {
  // 八边形的边必须包住伤员模型外接圆；内接八边形会低估80×40mm模型旋转后的范围。
  constexpr double pi = 3.14159265358979323846;
  const float radius =
      std::nextafter(static_cast<float>(0.5 * std::hypot(0.080, 0.040) / std::cos(pi / 8.0)),
                     std::numeric_limits<float>::infinity());
  std::vector<Point2D> result;
  for (int index = 0; index < 8; ++index) {
    const float angle = -kPi + static_cast<float>(index) * kPi / 4.0F;
    result.push_back({cx + radius * std::cos(angle), cy + radius * std::sin(angle)});
  }
  return result;
}

}  // 命名空间结束

bool validateCameraIntrinsics(const CameraIntrinsics& intrinsics) {
  return intrinsics.width > 0 && intrinsics.height > 0 && std::isfinite(intrinsics.fx) &&
         std::isfinite(intrinsics.fy) && std::isfinite(intrinsics.cx) &&
         std::isfinite(intrinsics.cy) && intrinsics.fx > 0.0F && intrinsics.fy > 0.0F;
}



GeometricObservation estimateGeometry(const Detection2D& detection, const CameraIntrinsics& intrinsics,
                                      const RigidTransform3D& camera_to_base,
                                      const GroundGeometryConfig& config) {
  // 纯RGB入口：相机射线与地面相交，不接受深度图或深度尺度。
  GeometricObservation result;
  result.class_id = detection.class_id;
  result.confidence = detection.confidence;
  result.bbox = detection.bbox;

  // 入口已选择本帧的RGB像素参考；此处不再接受第二套内参。
  const bool intrinsics_valid = validateCameraIntrinsics(intrinsics);
  const float box_width = detection.bbox.x_max - detection.bbox.x_min;
  const float box_height = detection.bbox.y_max - detection.bbox.y_min;
  const float center_x = 0.5F * (detection.bbox.x_min + detection.bbox.x_max);
  const float bottom_y = detection.bbox.y_max;

  // 这里先由框底部中心射线与地面求交，得到粗位置，仅供模型拟合初始化和搜索。后续当前RGB模型检查决定
  // 精细中心、姿态与围取候选；粗位置本身不代表模型已经通过。
  Point2D ground_center;
  if (intrinsics_valid && box_width > 0.0F && box_height > 0.0F &&
      projectGround(center_x, bottom_y, intrinsics, camera_to_base, &ground_center)) {
    result.geometry_valid = true;
    result.rgb_ground_anchor_valid = true;
    result.rgb_ground_anchor = ground_center;
    result.pose_x = ground_center.x;
    result.pose_y = ground_center.y;

    Point2D ground_left, ground_right;
    if (projectGround(detection.bbox.x_min, bottom_y, intrinsics, camera_to_base, &ground_left) &&
        projectGround(detection.bbox.x_max, bottom_y, intrinsics, camera_to_base, &ground_right)) {
      result.bbox_ground_width_m =
          std::hypot(ground_right.x - ground_left.x, ground_right.y - ground_left.y);
      result.bbox_ground_width_valid =
          std::isfinite(result.bbox_ground_width_m) && result.bbox_ground_width_m >= 0.0F;
    }
  }

  // 粗框宽度超过0.10m的判断由Vision统一测量和发布，Behavior消费bbox_oversized，避免两层使用不同阈值。
  result.bbox_oversized = result.bbox_ground_width_valid &&
                          result.bbox_ground_width_m > config.task_bbox_oversized_width_max_m;
  if (detection.class_id == 0U && result.bbox_oversized) {
    result.cluster_hint = true;
  }

  if (!result.geometry_valid) return result;

  if (detection.class_id == 2U) {
    result.footprint = conservativeCasualtyFootprint(result.pose_x, result.pose_y);
  } else {
    const float extent = result.cluster_hint && result.bbox_ground_width_valid
                             ? std::max(0.040F, result.bbox_ground_width_m)
                             : 0.040F;
    result.footprint = rectangle(result.pose_x, result.pose_y, 0.040F, extent, 0.0F);
  }
  return result;
}

}  // 命名空间结束 rescue_vision

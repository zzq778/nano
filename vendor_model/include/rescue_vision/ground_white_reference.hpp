// 模块职责：从近场白地面建立帧内唯一颜色参考，几何与危险色分支共用校正图；YOLO仍使用原图。
// 检测框排除物块，投地面范围排除远处背景；阴影亮度不直接改变颜色比例。
#pragma once
#include <rescue_vision/ground_geometry.hpp>
#include <rescue_vision/rgb_frame_view.hpp>
#include <algorithm>
#include <limits>
#include <array>
namespace rescue_vision {
struct GroundWhiteReference {
 bool valid{false};float red_gain{1},blue_gain{1};unsigned samples{0},cells{0};
};
inline GroundWhiteReference estimateGroundWhiteReference(const RgbFrameView& rgb,
 const std::vector<Detection2D>& detections,const CameraIntrinsics& k,const RigidTransform3D& tf){
 GroundWhiteReference out;
 if(!validateCameraIntrinsics(k)||!rgb.pixels||rgb.width!=k.width||rgb.height!=k.height||
    rgb.width>10000||rgb.height>10000||rgb.stride<rgb.width*3)return out;
 for(const auto& d:detections)
  if(!std::isfinite(d.bbox.x_min)||!std::isfinite(d.bbox.x_max)||!std::isfinite(d.bbox.y_min)||
     !std::isfinite(d.bbox.y_max)||d.bbox.x_min>=d.bbox.x_max||d.bbox.y_min>=d.bbox.y_max)return out;
 for(float v:tf.rotation)if(!std::isfinite(v))return out;
 for(float v:tf.translation)if(!std::isfinite(v))return out;
 std::vector<float> red,blue;std::array<unsigned,12> cells{};
 red.reserve(2048);blue.reserve(2048);
 for(int v=4;v<rgb.height;v+=8)for(int u=4;u<rgb.width;u+=8){
  Point2D p;if(!rayToGroundHeight(u,v,0,k,tf,&p)||p.x<.10F||p.x>.65F||std::abs(p.y)>.4F)continue;
  bool object=false;for(const auto& d:detections)
   if(u>=d.bbox.x_min-3&&u<=d.bbox.x_max+3&&v>=d.bbox.y_min-3&&v<=d.bbox.y_max+3){object=true;break;}
  if(object)continue;
  const auto* q=rgb.pixels+v*rgb.stride+u*3;
  const int peak=std::max({q[0],q[1],q[2]});
  if(q[1]<40||peak<80||peak>248)continue; // 黑线/暗物与过曝不作为白参考。
  red.push_back(float(q[0])/q[1]);blue.push_back(float(q[2])/q[1]);
  ++cells[std::min(2,v*3/rgb.height)*4+std::min(3,u*4/rgb.width)];
 }
 out.samples=unsigned(red.size());for(auto n:cells)out.cells+=n>=8;
 if(out.samples<64||out.cells<4)return out;
 const auto median=[](std::vector<float>& values){auto mid=values.begin()+values.size()/2;
  std::nth_element(values.begin(),mid,values.end());return *mid;};
 const float r=median(red),b=median(blue);
 if(!std::isfinite(r)||!std::isfinite(b)||r<.4F||r>2.5F||b<.4F||b>2.5F)return out;
 for(auto& q:red)q=std::abs(q-r);for(auto& q:blue)q=std::abs(q-b);
 if(median(red)>.08F*std::max(1.F,r)||median(blue)>.08F*std::max(1.F,b))return out; // 多种材质混成参考时拒绝。
 out.red_gain=1/r;out.blue_gain=1/b;out.valid=true;return out;
}
inline bool applyGroundWhiteReference(const RgbFrameView& rgb,const GroundWhiteReference& ref,
 std::vector<uint8_t>* output){
 if(!output||!ref.valid||!rgb.pixels||rgb.width<=0||rgb.height<=0||rgb.width>10000||rgb.height>10000||rgb.stride<rgb.width*3||
    !std::isfinite(ref.red_gain)||!std::isfinite(ref.blue_gain)||ref.red_gain<.4F||ref.red_gain>2.5F||
    ref.blue_gain<.4F||ref.blue_gain>2.5F)return false;
 output->resize(size_t(rgb.width)*rgb.height*3);std::array<uint8_t,256> r{},b{};
 for(unsigned i=0;i<256;++i){r[i]=uint8_t(std::min(255.F,std::round(i*ref.red_gain)));
                          b[i]=uint8_t(std::min(255.F,std::round(i*ref.blue_gain)));}
 // 固定行首与尺寸，像素内仅递增指针；保持逐字节LUT及原位行为。
 const int width=rgb.width,height=rgb.height,stride=rgb.stride;
 auto* destination=output->data();const auto* input=rgb.pixels;
 for(int v=0;v<height;++v){
  const auto* source=input+size_t(v)*stride;auto* target=destination+size_t(v)*width*3;
  for(int u=0;u<width;++u,source+=3,target+=3){target[0]=r[source[0]];target[1]=source[1];target[2]=b[source[2]];}
 }
 return true;
}
}

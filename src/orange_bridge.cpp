#include "orange_native.hpp"
#include <cmath>
#include <cstring>
#include <exception>
#include <stdexcept>
namespace {
double rounded1(double x){return std::nearbyint(x*10.)/10.;}
void points(cv::FileStorage& fs,const char* key,const std::vector<cv::Point>& ps){fs<<key<<"[";for(const auto& p:ps)fs<<"["<<p.x<<p.y<<"]";fs<<"]";}
}
extern "C" int orange_estimate_json(const unsigned char* data,int w,int h,int stride,char* out,int capacity){
 try{
  if(!data||!out||w<1||h<1||stride<w*3||capacity<1)throw std::invalid_argument("invalid orange ROI");
  cv::setNumThreads(1);
  const auto r=rv::estimate_orange(cv::Mat(h,w,CV_8UC3,const_cast<unsigned char*>(data),stride));
  cv::FileStorage fs("",cv::FileStorage::WRITE|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON);
  fs<<"status"<<r.status<<"reason"<<r.reason;
  if(r.has_bbox){
   fs<<"bbox_local"<<"["<<r.bbox.x<<r.bbox.y<<r.bbox.width<<r.bbox.height<<"]"<<"mask_area_px"<<r.mask_area_px;
   if(r.reason!="too small component")fs<<"n_profiles"<<r.n_profiles;
  }
  if(r.has_center)fs<<"center_local"<<"["<<r.center_local.x<<r.center_local.y<<"]";
  if(r.has_rims){
   const double cx=r.center_local.x;
   fs<<"rims_at_mid"<<"["<<rounded1(r.back_line[0]*cx+r.back_line[1])<<rounded1(r.front_line[0]*cx+r.front_line[1])<<"]";
   fs<<"top_depth_fraction"<<r.top_depth_fraction<<"edge_contrast"<<r.edge_contrast<<"scatter_px"<<r.scatter_px<<"survival"<<r.survival;
   fs<<"back_line"<<"["<<r.back_line[0]<<r.back_line[1]<<"]"<<"front_line"<<"["<<r.front_line[0]<<r.front_line[1]<<"]";
   points(fs,"back_points",r.back_points);points(fs,"front_points",r.front_points);
  }
  auto text=fs.releaseAndGetString();if(int(text.size()+1)>capacity)throw std::runtime_error("orange JSON buffer too small");
  std::memcpy(out,text.c_str(),text.size()+1);return int(text.size());
 }catch(const std::exception& e){if(out&&capacity>0){std::strncpy(out,e.what(),capacity-1);out[capacity-1]=0;}return -1;}
}

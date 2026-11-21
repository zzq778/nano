#include "v12_green.hpp"
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
namespace {
struct Session { v12::PreparedGreen frame; v12::Config cfg; };
void error(char* p,int n,const char* message) { if(p && n>0){std::strncpy(p,message,n-1);p[n-1]=0;} }
void point(cv::FileStorage& fs,const char* k,const cv::Point2d& p){fs<<k<<"["<<p.x<<p.y<<"]";}
void points(cv::FileStorage& fs,const char* k,const std::vector<cv::Point2d>& ps){
 if(ps.empty())return;
 fs<<k<<"[";for(auto& p:ps){fs<<"["<<p.x<<p.y<<"]";}fs<<"]";
}
}
extern "C" void* green_prepare(const unsigned char* data,int w,int h,int stride,const double* c,int gaussian3,char* err,int cap){
 try{
  if(!data||!c||w<1||h<1||stride<w*3)throw std::invalid_argument("invalid frame");
  cv::setNumThreads(1);std::unique_ptr<Session> s(new Session);
  s->frame=v12::prepare_green(cv::Mat(h,w,CV_8UC3,const_cast<unsigned char*>(data),stride),gaussian3!=0);
  s->cfg.back_expand_width=c[0];s->cfg.back_upper_width=c[1];s->cfg.min_depth_width=c[2];s->cfg.max_depth_width=c[3];
  s->cfg.minimum_contrast=c[4];s->cfg.fit_residual_fraction=c[5];s->cfg.minimum_profiles=int(c[6]);s->cfg.minimum_survival=c[7];
  return s.release();
 }catch(const std::exception& e){error(err,cap,e.what());return nullptr;}
}
extern "C" void green_release(void* p){delete static_cast<Session*>(p);}
extern "C" int green_local_json(void* context,const double* seed,char* out,int cap){
 try{
  if(!context||!seed||!out||cap<1)throw std::invalid_argument("invalid session or seed");
  auto* s=static_cast<Session*>(context);v12::Seed z{seed[0],seed[1],seed[2],seed[3],seed[4],seed[5]};
  if(z.x0<0||z.x1>=s->frame.mask.cols||z.y0<0||z.y1>=s->frame.mask.rows||z.x1<=z.x0||z.y1<=z.y0)throw std::invalid_argument("seed span out of frame or degenerate");
  const auto r=v12::refine_green_prepared(s->frame,z,s->cfg);
  cv::FileStorage fs("",cv::FileStorage::WRITE|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON);
  point(fs,"initial_center",r.initial_center);point(fs,"center",r.center);
  fs<<"status"<<r.status<<"reason"<<r.reason<<"n_profiles"<<r.n_profiles;
  if(r.has_lines){
   fs<<"back_line"<<"["<<r.back_line[0]<<r.back_line[1]<<"]";
   fs<<"front_line"<<"["<<r.front_line[0]<<r.front_line[1]<<"]";
   fs<<"median_contrast"<<"["<<r.median_contrast[0]<<r.median_contrast[1]<<"]";
   fs<<"rear_y"<<r.rear_y<<"front_y"<<r.front_y<<"survival"<<r.survival<<"x_span"<<"["<<z.x0<<z.x1<<"]";
  }
  points(fs,"back_points",r.back_points);points(fs,"front_points",r.front_points);
  points(fs,"debug_back",r.debug_back);points(fs,"debug_front",r.debug_front);
  auto json=fs.releaseAndGetString();if(int(json.size()+1)>cap)throw std::runtime_error("JSON capacity exhausted");
  std::memcpy(out,json.c_str(),json.size()+1);return int(json.size());
 }catch(const std::exception& e){error(out,cap,e.what());return -1;}
}

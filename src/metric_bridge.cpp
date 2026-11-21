// Pure RGB/known-size math imported from the audited handoff; no ROS runtime.
#include <rescue_vision/model_geometry.hpp>
#include <rescue_vision/ground_white_reference.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <limits>
namespace rv = rescue_vision;
namespace rescue_vision {
bool validateCameraIntrinsics(const CameraIntrinsics& k){
 return k.width>0&&k.height>0&&std::isfinite(k.fx)&&std::isfinite(k.fy)&&std::isfinite(k.cx)&&std::isfinite(k.cy)&&k.fx>0&&k.fy>0;
}
}
namespace {
struct Box {int cls;float confidence,x0,y0,x1,y1;};
struct Session {
 cv::Mat rgb;std::vector<uint8_t> balanced;rv::RgbFrameView view;
 rv::CameraIntrinsics k;rv::RigidTransform3D tf;rv::ModelFitScratch scratch;
 std::vector<rv::Detection2D> detections;rv::GroundWhiteReference white;
};
void error(char* out,int n,const char* message){if(out&&n>0){std::strncpy(out,message,n-1);out[n-1]=0;}}
void polygon(cv::FileStorage& fs,const char* key,const std::vector<rv::Point2D>& points){
 fs<<key<<"[";for(const auto& p:points)fs<<"["<<p.x<<p.y<<"]";fs<<"]";
}
}
extern "C" void* metric_prepare(const unsigned char* bgr,int w,int h,int stride,const double* k,
 const double* rt,const Box* boxes,int count,int balance,char* err,int capacity){
 try{
  if(!bgr||!k||!rt||w<1||h<1||w>10000||h>10000||stride<w*3||count<0||count>512||(count&&!boxes))throw std::invalid_argument("invalid metric frame");
  std::unique_ptr<Session> s(new Session);s->k={float(k[0]),float(k[1]),float(k[2]),float(k[3]),w,h};
  if(!rv::validateCameraIntrinsics(s->k))throw std::invalid_argument("invalid metric intrinsics");
  for(int i=0;i<12;++i)if(!std::isfinite(rt[i]))throw std::invalid_argument("nonfinite metric transform");
  for(int i=0;i<9;++i)s->tf.rotation[i]=rt[i];
  for(int i=0;i<3;++i)s->tf.translation[i]=rt[9+i];
  // The caller provides ground-referenced camera height, normalized exactly once.
  if(s->tf.translation[2]<=0)throw std::invalid_argument("camera must be above ground");
  for(int i=0;i<count;++i){const auto& b=boxes[i];rv::Detection2D d;d.class_id=b.cls;d.confidence=b.confidence;d.bbox={b.x0,b.y0,b.x1,b.y1};s->detections.push_back(d);}
  cv::cvtColor(cv::Mat(h,w,CV_8UC3,const_cast<unsigned char*>(bgr),stride),s->rgb,cv::COLOR_BGR2RGB);
  s->view={s->rgb.ptr<unsigned char>(),w,h,int(s->rgb.step)};
  if(balance){s->white=rv::estimateGroundWhiteReference(s->view,s->detections,s->k,s->tf);
   if(rv::applyGroundWhiteReference(s->view,s->white,&s->balanced))s->view={s->balanced.data(),w,h,w*3};}
  return s.release();
 }catch(const std::exception& e){error(err,capacity,e.what());return nullptr;}
}
extern "C" int metric_fit_json(void* context,int index,double budget_ms,char* out,int capacity){
 try{
  auto* s=static_cast<Session*>(context);if(!s||index<0||index>=int(s->detections.size())||!std::isfinite(budget_ms)||budget_ms<0)throw std::invalid_argument("invalid metric task");
  rv::GeometricObservation coarse;rv::Point2D anchor;const auto& box=s->detections[index].bbox;
  coarse.geometry_valid=rv::rayToGroundHeight((box.x_min+box.x_max)*.5F,box.y_max,0,s->k,s->tf,&anchor);
  if(coarse.geometry_valid){coarse.pose_x=anchor.x;coarse.pose_y=anchor.y;}
  rv::ModelFitOptions options;options.scratch=&s->scratch;
  if(budget_ms>0)options.deadline=std::chrono::steady_clock::now()+std::chrono::microseconds(int64_t(budget_ms*1000));
  // No assumed occlusion ordering or remembered poses; fresh RGB evidence each time.
  auto r=rv::fitGroundModel(s->detections[index],s->view,s->k,s->tf,coarse,options);
  cv::FileStorage fs("",cv::FileStorage::WRITE|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON);
  fs<<"status"<<int(r.status)<<"reason_code"<<int(r.reason)<<"x"<<r.x<<"y"<<r.y<<"yaw_rad"<<r.yaw<<"error_px"<<r.error_px;
  fs<<"evaluations"<<int(r.evaluations)<<"visible_fraction"<<r.visible_fraction<<"white_reference_valid"<<int(s->white.valid);
  fs<<"white_red_gain"<<s->white.red_gain<<"white_blue_gain"<<s->white.blue_gain;
  if(r.best_effort_available)fs<<"best_effort"<<"{"<<"x"<<r.best_effort.x<<"y"<<r.best_effort.y<<"yaw_rad"<<r.best_effort.yaw<<"cost"<<r.best_effort_cost<<"error_px"<<r.best_effort.error_px<<"}";
  polygon(fs,"footprint",r.footprint);fs<<"candidates"<<"[";
  for(const auto& c:r.candidates){fs<<"{"<<"x"<<c.x<<"y"<<c.y<<"yaw_rad"<<c.yaw<<"error_px"<<c.error_px;polygon(fs,"footprint",c.footprint);fs<<"}";}fs<<"]";
  const auto text=fs.releaseAndGetString();if(int(text.size()+1)>capacity)throw std::runtime_error("metric output too large");
  std::memcpy(out,text.c_str(),text.size()+1);return int(text.size());
 }catch(const std::exception& e){error(out,capacity,e.what());return -1;}
}
extern "C" void metric_release(void* context){delete static_cast<Session*>(context);}

// Same 19 latent orientations and five box-edge iterations as shape_center.py.
// Orientation is a nuisance variable for locating the cube; it is never exported.
extern "C" int metric_cube_json(void* context,int index,char* out,int capacity){
 try{
  auto* s=static_cast<Session*>(context);if(!s||index<0||index>=int(s->detections.size()))throw std::invalid_argument("invalid cube task");
  const auto b=s->detections[index].bbox;const auto& k=s->k;const auto& r=s->tf.rotation;const auto& t=s->tf.translation;
  struct Fit {double error,x,y;};std::vector<Fit> fits;rv::Point2D initial;
  if(b.x_min>=2&&b.y_min>=2&&b.x_max<=k.width-3&&b.y_max<=k.height-3&&b.x_max-b.x_min>=8&&b.y_max-b.y_min>=8&&
    rv::rayToGroundHeight((b.x_min+b.x_max)*.5,(b.y_min+b.y_max)*.5,.02,k,s->tf,&initial)){
   const double observed[4]={b.x_min,b.x_max,b.y_min,b.y_max};
   for(int yi=0;yi<19;++yi){
    const double yaw=yi*3.14159265358979323846/36,co=std::cos(yaw),si=std::sin(yaw);
    double offset[8][3]={};int n=0;
    for(double z:{0.,.04})for(double y:{-.02,.02})for(double x:{-.02,.02}){
     const double point[3]={co*x-si*y-t[0],si*x+co*y-t[1],z-t[2]};
     for(int j=0;j<3;++j)for(int a=0;a<3;++a)offset[n][j]+=point[a]*r[a*3+j];
     ++n;
    }
    double x=initial.x,y=initial.y,pixels[8][2];bool valid=true;
    auto project=[&](){for(int i=0;i<8;++i){const double depth=offset[i][2]+x*r[2]+y*r[5];if(depth<=.01)return false;
      pixels[i][0]=k.fx*(offset[i][0]+x*r[0]+y*r[3])/depth+k.cx;
      pixels[i][1]=k.fy*(offset[i][1]+x*r[1]+y*r[4])/depth+k.cy;}return true;};
    for(int iteration=0;iteration<5;++iteration){
     if(!project()){valid=false;break;}
     int ids[4]={0,0,0,0};for(int i=1;i<8;++i){if(pixels[i][0]<pixels[ids[0]][0])ids[0]=i;if(pixels[i][0]>pixels[ids[1]][0])ids[1]=i;if(pixels[i][1]<pixels[ids[2]][1])ids[2]=i;if(pixels[i][1]>pixels[ids[3]][1])ids[3]=i;}
     double aa=0,ab=0,bb=0,av=0,bv=0;
     for(int side=0;side<4;++side){int axis=side<2?0:1,i=ids[side];double q=(observed[side]-(axis==0?k.cx:k.cy))/(axis==0?k.fx:k.fy);
      double a=r[axis]-q*r[2],b2=r[3+axis]-q*r[5],v=q*offset[i][2]-offset[i][axis];aa+=a*a;ab+=a*b2;bb+=b2*b2;av+=a*v;bv+=b2*v;}
     const double det=aa*bb-ab*ab;if(det<=1e-12){valid=false;break;}
     const double nx=(bb*av-ab*bv)/det,ny=(aa*bv-ab*av)/det,delta=std::hypot(nx-x,ny-y);x=nx;y=ny;if(delta<1e-6)break;
    }
    if(!valid||!project()||std::hypot(x-t[0],y-t[1])>=4)continue;
    double bounds[4]={pixels[0][0],pixels[0][0],pixels[0][1],pixels[0][1]};
    for(int i=1;i<8;++i){bounds[0]=std::min(bounds[0],pixels[i][0]);bounds[1]=std::max(bounds[1],pixels[i][0]);bounds[2]=std::min(bounds[2],pixels[i][1]);bounds[3]=std::max(bounds[3],pixels[i][1]);}
    double error=0;for(int side=0;side<4;++side)error+=(bounds[side]-observed[side])*(bounds[side]-observed[side]);fits.push_back({std::sqrt(error/4),x,y});
   }
  }
  double best=std::numeric_limits<double>::infinity(),x=0,y=0,radius=0;int accepted=0;
  for(const auto& f:fits)best=std::min(best,f.error);
  bool valid=best<=std::max(3.,.06*std::max(b.x_max-b.x_min,b.y_max-b.y_min));
  if(valid){for(const auto& f:fits)if(f.error<=best+1){x+=f.x;y+=f.y;++accepted;}x/=accepted;y/=accepted;
    for(const auto& f:fits)if(f.error<=best+1)radius=std::max(radius,std::hypot(f.x-x,f.y-y));}
  cv::FileStorage fs("",cv::FileStorage::WRITE|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON);fs<<"valid"<<int(valid);
  if(valid)fs<<"center_m"<<"["<<x<<y<<"]"<<"candidate_radius_mm"<<radius*1000<<"bbox_rms_px"<<best<<"latent_yaw_samples"<<int(fits.size());
  const auto text=fs.releaseAndGetString();if(int(text.size()+1)>capacity)throw std::runtime_error("cube output too large");std::memcpy(out,text.c_str(),text.size()+1);return int(text.size());
 }catch(const std::exception& e){error(out,capacity,e.what());return -1;}
}

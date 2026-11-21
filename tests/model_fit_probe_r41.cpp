// Host-only fixture runner for the pure C++ RGB known-size fitting core.
// This deliberately does not use OpenCV, CUDA, the Jetson or any camera.
#include <rescue_vision/model_geometry.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <string>
#include <cmath>
using namespace rescue_vision;
int main(int argc,char** argv){
 if(argc!=2){std::cerr<<"usage: model_fit_probe manifest.txt\n";return 2;}
 std::ifstream in(argv[1]);if(!in){std::cerr<<"missing manifest\n";return 2;}
 CameraIntrinsics k;RigidTransform3D tf;
 in>>k.width>>k.height>>k.fx>>k.fy>>k.cx>>k.cy;
 for(float& r:tf.rotation)in>>r;
 for(float& t:tf.translation)in>>t;
 int count=0;in>>count;
 if(!in||count<1||count>256){std::cerr<<"invalid header\n";return 3;}
 std::cout<<std::setprecision(9);
 ModelFitScratch scratch;
 for(int i=0;i<count;++i){
  std::string source;
  int cls;float x0,y0,x1,y1;
  in>>source>>cls>>x0>>y0>>x1>>y1;
  if(!in||cls<0||cls>3){std::cerr<<"invalid case\n";return 3;}
  std::ifstream frame(source,std::ios::binary);
  std::vector<uint8_t> rgb(size_t(k.width)*size_t(k.height)*3);
  if(!frame.read(reinterpret_cast<char*>(rgb.data()),rgb.size())){
   std::cerr<<"missing/truncated frame: "<<source<<"\n";return 4;
  }
  RgbFrameView view{rgb.data(),k.width,k.height,k.width*3};
  Detection2D d;d.class_id=static_cast<uint8_t>(cls);d.confidence=.9F;
  d.bbox={x0,y0,x1,y1};
  GeometricObservation coarse;Point2D anchor;
  coarse.geometry_valid=rayToGroundHeight((x0+x1)*.5F,y1,0,k,tf,&anchor);
  if(coarse.geometry_valid){coarse.pose_x=anchor.x;coarse.pose_y=anchor.y;}
  ModelFitOptions options;options.scratch=&scratch;
  auto start=std::chrono::steady_clock::now();
  const auto r=fitGroundModel(d,view,k,tf,coarse,options);
  auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  std::cout<<i<<' '<<int(coarse.geometry_valid)<<' '<<int(r.status)<<' '<<int(r.reason)
   <<' '<<r.evaluations<<' '<<r.x<<' '<<r.y<<' '<<r.yaw<<' '<<r.error_px
   <<' '<<r.visible_fraction<<' '<<r.best_effort_available<<' '<<r.best_effort_cost
   <<' '<<r.best_effort.x<<' '<<r.best_effort.y<<' '<<r.best_effort.yaw
   <<' '<<r.candidates.size()<<' '<<r.footprint.size()<<' '<<elapsed<<'\n';
 }
}

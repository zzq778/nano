// Generic current YOLO frontend only: no legacy geometry, ROS, scheduler, UART.
#include <rescue_vision/tensorrt_engine.hpp>
#include <rescue_vision/detection_postprocessor.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <exception>
#include <memory>
#include <stdexcept>
using namespace rescue_vision;
struct Detector{TensorRtEngine engine;std::array<float,6> thresholds{{.5F,.5F,.2F,.35F,.35F,.35F}};float nms=.45F;double last_ms[3]={0,0,0};};
extern "C" int detector_last_timings(void* opaque,double* out){
 if(!opaque||!out)return -1;
 auto* d=static_cast<Detector*>(opaque);
 for(int i=0;i<3;++i)out[i]=d->last_ms[i];
 return 0;
}
struct DetectionRecord{int cls;float confidence,x0,y0,x1,y1;};
extern "C" void* detector_create(const char* path,char* error,int error_capacity){
 if(!error||error_capacity<1)return nullptr;
 try{
  if(!path)throw std::invalid_argument("null model path");
  std::unique_ptr<Detector> detector(new Detector);std::string message;
  if(!detector->engine.load(path,&message)){std::strncpy(error,message.c_str(),error_capacity-1);error[error_capacity-1]=0;return nullptr;}
  cv::setNumThreads(1);return detector.release();
 }catch(const std::exception& e){std::strncpy(error,e.what(),error_capacity-1);error[error_capacity-1]=0;return nullptr;}
}
extern "C" void detector_destroy(void* opaque){delete static_cast<Detector*>(opaque);}
extern "C" int detector_configure(void* opaque,const float* thresholds,float nms){
 if(!opaque||!thresholds||!std::isfinite(nms)||nms<=0||nms>1)return -1;
 auto* detector=static_cast<Detector*>(opaque);
 for(int i=0;i<6;++i){if(!std::isfinite(thresholds[i])||thresholds[i]<0||thresholds[i]>1)return -1;}
 for(int i=0;i<6;++i){detector->thresholds[i]=thresholds[i];}
 detector->nms=nms;return 0;
}
// Reused per-thread staging: the native context is documented as single-thread
// owned, and re-allocating a 1.23M-element float buffer every frame is measurable
// on the Nano. Contents are fully rewritten on each call.
struct DetectorBuffers {
 cv::Mat resized, padded, rgb;
 std::vector<cv::Mat> planes;
 std::vector<float> input, output;
 std::vector<RawDetection> raw;
 std::vector<Detection2D> restored;
};
static int run_detector(void* opaque,const unsigned char* pixels,int width,int height,int stride,DetectionRecord* records,int capacity,char* error,int error_capacity){
 if(!opaque||!pixels||!records||width<1||height<1||stride<width*3||capacity<1||!error||error_capacity<1)return -4;
 auto* detector=static_cast<Detector*>(opaque);const auto& info=detector->engine.info();
 const auto started=std::chrono::steady_clock::now();
 static thread_local DetectorBuffers buffers;DetectorBuffers& b=buffers;
 cv::Mat bgr(height,width,CV_8UC3,const_cast<unsigned char*>(pixels),stride);
 LetterboxTransform letterbox;
 if(!makeLetterboxTransform(width,height,info.input_width,info.input_height,&letterbox))return -1;
 // Resize before the color conversion: identical pixels, one third of the work.
 cv::resize(bgr,b.resized,cv::Size(letterbox.resized_width,letterbox.resized_height));
 int left=int(letterbox.pad_x),top=int(letterbox.pad_y);
 cv::copyMakeBorder(b.resized,b.padded,top,info.input_height-letterbox.resized_height-top,left,info.input_width-letterbox.resized_width-left,cv::BORDER_CONSTANT,cv::Scalar(114,114,114));
 cv::cvtColor(b.padded,b.rgb,cv::COLOR_BGR2RGB);
 size_t plane=size_t(info.input_width)*info.input_height;
 if(b.input.size()!=3*plane)b.input.resize(3*plane);
 if(b.planes.size()!=3)b.planes.resize(3);
 cv::split(b.rgb,b.planes);
 for(int c=0;c<3;++c){
  cv::Mat dst(info.input_height,info.input_width,CV_32F,b.input.data()+c*plane);
  b.planes[c].convertTo(dst,CV_32F,1.0/255.0);
 }
 const auto preprocessed=std::chrono::steady_clock::now();
 std::string message;
 if(!detector->engine.infer(b.input.data(),b.input.size(),&b.output,&message)){
  std::strncpy(error,message.c_str(),error_capacity-1);error[error_capacity-1]=0;return -2;
 }
 const auto inferred=std::chrono::steady_clock::now();
 if(!YoloRawAdapter::adapt(b.output.data(),b.output.size(),info.output.dimensions,detector->thresholds,&b.raw,&message)){
  std::strncpy(error,message.c_str(),error_capacity-1);error[error_capacity-1]=0;return -2;
 }
 b.restored.clear();
 for(const auto& detection:b.raw){
  Detection2D restored;
  if(restoreDetection(detection,letterbox,&restored))b.restored.push_back(restored);
 }
 auto boxes=classWiseNms(b.restored,detector->nms);
 if(int(boxes.size())>capacity)return -3;
 for(size_t i=0;i<boxes.size();++i){auto& d=boxes[i];records[i]={int(d.class_id),d.confidence,d.bbox.x_min,d.bbox.y_min,d.bbox.x_max,d.bbox.y_max};}
 const auto done=std::chrono::steady_clock::now();
 const auto milliseconds=[](std::chrono::steady_clock::duration elapsed){
  return std::chrono::duration<double,std::milli>(elapsed).count();
 };
 detector->last_ms[0]=milliseconds(preprocessed-started);
 detector->last_ms[1]=milliseconds(inferred-preprocessed);
 detector->last_ms[2]=milliseconds(done-inferred);
 return int(boxes.size());
}
extern "C" int detector_run(void* p,const unsigned char* data,int w,int h,int stride,DetectionRecord* boxes,int capacity,char* err,int n){
 try{return run_detector(p,data,w,h,stride,boxes,capacity,err,n);}
 catch(const std::exception& e){if(err&&n>0){std::strncpy(err,e.what(),n-1);err[n-1]=0;}return -5;}
}

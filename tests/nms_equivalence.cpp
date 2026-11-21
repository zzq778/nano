#include <rescue_vision/detection_postprocessor.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using namespace rescue_vision;
std::vector<Detection2D> reference(const std::vector<Detection2D>& detections,float iou){
 std::vector<size_t> order(detections.size());for(size_t i=0;i<order.size();++i)order[i]=i;
 std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return detections[a].confidence>detections[b].confidence;});
 std::vector<Detection2D> kept;
 for(auto i:order){
  auto& d=detections[i];bool blocked=false;
  for(auto& k:kept){if(d.class_id==k.class_id && intersectionOverUnion(d.bbox,k.bbox)>iou){blocked=true;break;}}
  if(!blocked)kept.push_back(d);
 }
 return kept;
}

int main(){
 std::mt19937 gen(42);
 std::uniform_real_distribution<float> coord(0,640),score(0,1);
 int validations=0;double fastMs=0,slowMs=0;
 for(int scenario=0;scenario<50;++scenario){
  int n=(scenario%5==0)?5000:200+scenario*11;
  std::vector<Detection2D> ds;
  for(int i=0;i<n;++i){
   float x=coord(gen),y=coord(gen),w=20+coord(gen)*.1f,h=20+coord(gen)*.1f;
   uint8_t cls=(i%12==0)?uint8_t(200):uint8_t(i%6);
   ds.push_back({cls,score(gen),{x,y,x+w,y+h}});
  }
  float iou=(scenario%2)?0.45f:0.3f;
  auto begin=std::chrono::steady_clock::now();auto a=reference(ds,iou);auto split=std::chrono::steady_clock::now();auto b=classWiseNms(ds,iou);auto end=std::chrono::steady_clock::now();
  fastMs+=std::chrono::duration<double,std::milli>(end-split).count();
  slowMs+=std::chrono::duration<double,std::milli>(split-begin).count();
  if(a.size()!=b.size()){std::printf("different count at %d\n",scenario);return 1;}
  for(size_t k=0;k<a.size();++k){
   if(a[k].class_id!=b[k].class_id || a[k].confidence!=b[k].confidence || std::memcmp(&a[k].bbox,&b[k].bbox,sizeof(BoundingBox2D))){
    std::printf("different output at scenario=%d index=%zu\n",scenario,k);return 2;
   }
  }
  ++validations;
 }
 std::printf("NMS_PARITY_PASS scenarios=%d reference_ms=%.3f optimized_ms=%.3f speedup=%.2fx\n",validations,slowMs,fastMs,slowMs/fastMs);
 return 0;
}

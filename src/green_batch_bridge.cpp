#include "v12_green.hpp"
#include <cstring>
#include <exception>

static void point(cv::FileStorage& fs, const char* key, const cv::Point2d& p) {
    fs << key << "[" << p.x << p.y << "]";
}
static void points(cv::FileStorage& fs, const char* key, const std::vector<cv::Point2d>& ps) {
    if (ps.empty()) return;
    fs << key << "[";
    for (const auto& p : ps) fs << "[" << p.x << p.y << "]";
    fs << "]";
}
extern "C" int v12_refine_json(const unsigned char* data, int w, int h, int stride,
                              const double* seeds, int n, const double* config, int gaussian3,
                              char* output, int capacity) {
    try {
        if (!data || !output || n < 0 || n > 512 || capacity < 1) return -1;
        cv::setNumThreads(1);
        cv::Mat image(h, w, CV_8UC3, const_cast<unsigned char*>(data), stride);
        std::vector<v12::Seed> ss;
        for (int i=0; i<n; ++i) {
            const double* s=seeds+6*i;
            ss.push_back({s[0],s[1],s[2],s[3],s[4],s[5]});
        }
        v12::Config cfg;
        cfg.back_expand_width=config[0]; cfg.back_upper_width=config[1];
        cfg.min_depth_width=config[2]; cfg.max_depth_width=config[3];
        cfg.minimum_contrast=config[4]; cfg.fit_residual_fraction=config[5];
        cfg.minimum_profiles=int(config[6]); cfg.minimum_survival=config[7];
        auto results=v12::refine_green_group(image,ss,cfg,gaussian3!=0);
        cv::FileStorage fs("",cv::FileStorage::WRITE|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON);
        fs << "results" << "[";
        for (int i=0; i<n; ++i) {
            const auto& r=results[i]; fs << "{";
            point(fs,"initial_center",r.initial_center); point(fs,"center",r.center);
            fs << "status" << r.status << "reason" << r.reason << "n_profiles" << r.n_profiles;
            if(r.has_proposed) point(fs,"proposed_center",r.proposed_center);
            if(r.conflicting_index>=0) fs << "conflicting_candidate_index" << r.conflicting_index;
            if(r.has_lines) {
                fs << "back_line" << "[" << r.back_line[0] << r.back_line[1] << "]";
                fs << "front_line" << "[" << r.front_line[0] << r.front_line[1] << "]";
                fs << "median_contrast" << "[" << r.median_contrast[0] << r.median_contrast[1] << "]";
                fs << "rear_y" << r.rear_y << "front_y" << r.front_y << "survival" << r.survival;
                fs << "x_span" << "[" << ss[i].x0 << ss[i].x1 << "]";
            }
            points(fs,"back_points",r.back_points); points(fs,"front_points",r.front_points);
            points(fs,"debug_back",r.debug_back); points(fs,"debug_front",r.debug_front);
            fs << "}";
        }
        fs << "]"; auto str=fs.releaseAndGetString();
        if(int(str.size()+1)>capacity) return -2;
        std::memcpy(output,str.c_str(),str.size()+1); return int(str.size());
    } catch(const std::exception& e) {
        if(output && capacity>0) {std::strncpy(output,e.what(),capacity-1);output[capacity-1]=0;}
        return -3;
    }
}

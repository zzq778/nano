#include <cstring>
#include <exception>
#include <rescue_vision/topface.hpp>
#include <stdexcept>

// Fixed C ABI: pointers remain owned by the caller; no JSON or heap ownership
// crosses the boundary. Inclusive observed spans are not full-object boxes.
struct TopFaceRecord {
  double x, y;
  int pixels;
  int component_x, component_y, component_width, component_height;
  int xmin, xmax, ymin, ymax;
  int split, isolated_fallback;
};

extern "C" int topface_detect(const unsigned char* data, int width, int height,
                              int stride, TopFaceRecord* output, int capacity,
                              char* error, int error_capacity) {
  try {
    if (!data || width < 1 || height < 1 || width > 10000 || height > 10000 ||
        stride < width * 3 || !output || capacity < 1)
      throw std::invalid_argument("invalid topface image or output buffer");
    const cv::Mat bgr(height, width, CV_8UC3, const_cast<unsigned char*>(data),
                      stride);
    const auto faces = rescue_vision::detectTopFaces(bgr);
    if (faces.size() > static_cast<size_t>(capacity))
      throw std::length_error(
          "topface output capacity exceeded; no partial result returned");
    for (size_t i = 0; i < faces.size(); ++i) {
      const auto& p = faces[i];
      output[i] = {
          p.center.x,
          p.center.y,
          p.pixels,
          p.component.x,
          p.component.y,
          p.component.width,
          p.component.height,
          p.min_pixel.x,
          p.max_pixel.x,
          p.min_pixel.y,
          p.max_pixel.y,
          p.split ? 1 : 0,
          p.kind == rescue_vision::TopFaceKind::IsolatedColorFallback ? 1 : 0};
    }
    return static_cast<int>(faces.size());
  } catch (const std::exception& exception) {
    if (error && error_capacity > 0) {
      std::strncpy(error, exception.what(), error_capacity - 1);
      error[error_capacity - 1] = 0;
    }
    return -1;
  }
}

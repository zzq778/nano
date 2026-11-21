// Single-owner Astra SDK boundary. This library is used only in the capture process.
#include <OpenNI.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
bool sdk_owned = false;
void error(char* output, int capacity, const char* message) {
  if (output && capacity > 0) {
    std::strncpy(output, message, size_t(capacity - 1));
    output[capacity - 1] = 0;
  }
}
void checked(openni::Status status, const char* operation) {
  if (status != openni::STATUS_OK)
    throw std::runtime_error(std::string(operation) + ": " + openni::OpenNI::getExtendedError());
}
struct Session {
  openni::Device device;
  openni::VideoStream color;
  bool initialized = false;
  ~Session() {
    if (color.isValid()) { color.stop(); color.destroy(); }
    if (device.isValid()) device.close();
    if (initialized) { openni::OpenNI::shutdown(); sdk_owned = false; }
  }
};
}

extern "C" void* astra_open(const char* requested_uri, int width, int height,
                             char* output_error, int error_capacity) {
  try {
    if (!requested_uri || width != 640 || height != 480)
      throw std::invalid_argument("Astra input requires explicit 640x480 RGB");
    const std::string requested(requested_uri);
    if (requested != "astra" && requested.compare(0, 10, "2bc5/0402@") != 0)
      throw std::invalid_argument("expected astra or an explicit Astra device URI");
    if (sdk_owned) throw std::runtime_error("one Astra session is permitted per capture process");
    std::unique_ptr<Session> session(new Session);
    if (setenv("OPENNI2_REDIST", RESCUE_ASTRA_REDIST, 1) != 0)
      throw std::runtime_error("cannot set the explicit SDK driver directory");
    checked(openni::OpenNI::initialize(), "SDK initialization");
    session->initialized = true; sdk_owned = true;
    openni::Array<openni::DeviceInfo> devices;
    openni::OpenNI::enumerateDevices(&devices);
    std::string selected; unsigned matches = 0;
    for (int i = 0; i < devices.getSize(); ++i) {
      const std::string uri(devices[i].getUri());
      if (uri.compare(0, 10, "2bc5/0402@") != 0) continue;
      if (requested != "astra" && uri != requested) continue;
      selected = uri; ++matches;
    }
    if (matches != 1) throw std::runtime_error("expected exactly one matching Astra device");
    checked(session->device.open(selected.c_str()), "device open");
    const auto* sensor = session->device.getSensorInfo(openni::SENSOR_COLOR);
    if (!sensor) throw std::runtime_error("Astra RGB sensor unavailable");
    const auto& modes = sensor->getSupportedVideoModes(); int selected_mode = -1;
    for (int i = 0; i < modes.getSize(); ++i) {
      const auto& mode = modes[i];
      if (mode.getResolutionX() == width && mode.getResolutionY() == height &&
          mode.getFps() == 30 && mode.getPixelFormat() == openni::PIXEL_FORMAT_RGB888)
        selected_mode = i;
    }
    if (selected_mode < 0) throw std::runtime_error("640x480 RGB888 at 30Hz unavailable");
    checked(session->color.create(session->device, openni::SENSOR_COLOR), "RGB stream create");
    checked(session->color.setVideoMode(modes[selected_mode]), "RGB mode");
    checked(session->color.start(), "RGB stream start");
    checked(session->color.setMirroringEnabled(false), "disable mirroring");
    if (session->color.getMirroringEnabled()) throw std::runtime_error("RGB mirroring remains enabled");
    return session.release();
  } catch (const std::exception& exception) {
    error(output_error, error_capacity, exception.what()); return nullptr;
  }
}

extern "C" int astra_read(void* context, unsigned char* bgr, int capacity, int timeout_ms,
                           unsigned long long* device_timestamp, int* device_frame_index,
                           char* output_error, int error_capacity) {
  try {
    if (!context || !bgr || capacity < 640 * 480 * 3 || timeout_ms < 1 || timeout_ms > 1000 ||
        !device_timestamp || !device_frame_index)
      throw std::invalid_argument("invalid Astra read buffer or deadline");
    auto& session = *static_cast<Session*>(context);
    openni::VideoStream* streams[] = {&session.color}; int ready = -1;
    const auto status = openni::OpenNI::waitForAnyStream(streams, 1, &ready, timeout_ms);
    if (status == openni::STATUS_TIME_OUT) return 0;
    checked(status, "RGB wait");
    openni::VideoFrameRef frame;
    checked(session.color.readFrame(&frame), "RGB read");
    if (!frame.isValid() || frame.getWidth() != 640 || frame.getHeight() != 480 ||
        frame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_RGB888 ||
        frame.getCroppingEnabled() || frame.getStrideInBytes() < 640 * 3 ||
        frame.getDataSize() < frame.getStrideInBytes() * 479 + 640 * 3)
      throw std::runtime_error("unexpected Astra frame format or geometry");
    const auto* pixels = static_cast<const unsigned char*>(frame.getData());
    for (int y = 0; y < 480; ++y) {
      const auto* source = pixels + y * frame.getStrideInBytes();
      auto* destination = bgr + y * 640 * 3;
      for (int x = 0; x < 640; ++x) {
        destination[x * 3] = source[x * 3 + 2];
        destination[x * 3 + 1] = source[x * 3 + 1];
        destination[x * 3 + 2] = source[x * 3];
      }
    }
    *device_timestamp = frame.getTimestamp(); *device_frame_index = frame.getFrameIndex();
    return 1;
  } catch (const std::exception& exception) {
    error(output_error, error_capacity, exception.what()); return -1;
  }
}

extern "C" void astra_close(void* context) { delete static_cast<Session*>(context); }

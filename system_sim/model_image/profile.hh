#pragma once
#include "device.hh"

namespace mlx::model_image {
struct Profile {
  clocked_device::Options options;
  Json::Value effective;
};
// Empty/null selects the historical one-PE demonstration. Explicit profiles
// use the backend parsers and record every effective resource/timing field.
Profile parse_profile(const Json::Value &value);
}

#pragma once
#include <json/json.h>
namespace mlx::event_schedule {
// Timing-only, finite-resource calendar. Does not produce numerical tensors.
Json::Value simulate(const Json::Value &program);
}

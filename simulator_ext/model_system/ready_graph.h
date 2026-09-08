#pragma once
#include "model_executor.h"

namespace mlx::model_system {
// Whole-op dependencies with real shared matrix/vector resources. This is
// not a partial-tile CDC scheduler, a real CPU, or a performance certificate.
Json::Value execute_ready_graph(const Json::Value &program,const Json::Value &options,
                               const std::filesystem::path &output_directory);
}

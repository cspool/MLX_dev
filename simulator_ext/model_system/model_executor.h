#pragma once
#include "physical_memory.h"
#include <filesystem>

namespace mlx::model_system {
// Serial graph control with a shared physical address space/clock. Each
// backend retains its real bounded issue/response behavior. Cross-operator
// compute overlap, CPU instruction fetch, and Chipyard integration are pending.
Json::Value execute(const Json::Value &program,const Json::Value &system_options,
                    const std::filesystem::path &output_directory);
} // namespace mlx::model_system

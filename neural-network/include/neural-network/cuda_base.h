#pragma once

#ifndef BUILD_CUDA_SUPPORT
#error cuda_base.h cannot be included without BUILD_CUDA_SUPPORT
#endif // !BUILD_CUDA_SUPPORT

#include <cstdint>

namespace nn {

constexpr uint32_t CUDA_THREADS = 256;
constexpr uint32_t CUDA_MINIMUM = 100'000;

} // namespace nn

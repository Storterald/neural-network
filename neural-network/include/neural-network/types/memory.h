#pragma once

/* The Memory.h header allows for memory management with runtime chosen location.
 *
 * The over-head cost of the wrappers is nearly 100x less than the cost of
 * managed memory.
 *
 * The exposed classes are not multithread safe, the responsibility is of the
 * caller.
 *
 */

#include <type_traits>

#ifdef BUILD_CUDA_SUPPORT
#include "_memory_cuda.h"
#else // BUILD_CUDA_SUPPORT
#include "_memory_cpu.h"
#endif // BUILD_CUDA_SUPPORT

namespace nn {

template <typename T>
using view = span<std::add_const_t<T>>;

} // namespace nn
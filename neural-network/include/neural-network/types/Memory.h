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

#ifdef BUILD_CUDA_SUPPORT
#include "CudaMemory.h"
#else // BUILD_CUDA_SUPPORT
#include "BaseMemory.h"
#endif // BUILD_CUDA_SUPPORT

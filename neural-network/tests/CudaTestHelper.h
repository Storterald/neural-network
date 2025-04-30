#pragma once

#include <neural-network/types/buf.h>

namespace helper {

        void access_values(uint32_t size, const float *data);
        bool check_values(uint32_t size, const float *data, float v);
        void set_values(uint32_t size, float *data, float v);

} // namespace helper

#pragma once

#include <neural-network/types/Data.h>

namespace Helper {

        void access_values(uint32_t size, const float *data);
        bool check_values(uint32_t size, const float *data, float v);

}

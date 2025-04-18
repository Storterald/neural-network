#pragma once

#include <types/Data.h>

namespace CUDATest {

        void access_values(const Data &data);
        bool check_values(const Data &data, float v);
}

#pragma once

#include "../src/types/Data.h"

namespace CUDATest {

        void accessValues(const Data &data);
        bool checkValues(const Data &data, float v);
}

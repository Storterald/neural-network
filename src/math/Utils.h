#pragma once

#include "Vector.h"

namespace Utils {

        [[nodiscard]] Vector min(const Vector &vec, const Vector &other);
        [[nodiscard]] Vector clamp(const Vector &vec, float min, float max);
        [[nodiscard]] float maxElement(const Vector &vec);
        [[nodiscard]] uint32_t maxElementIndex(const Vector &vec);

} // namespace Utils
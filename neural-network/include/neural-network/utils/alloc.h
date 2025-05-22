#pragma once

#include <new>

namespace nn::memory {

template <typename T>
inline T *alloc(size_t count)
{
        // MSVC bug, new (std::align_val_t(64)) T[count] causes error C2956.
        return (T *)::operator new (sizeof(T) * count, std::align_val_t(64));
}

template <typename T>
inline T *calloc(size_t count)
{
        auto ptr = (T *)::operator new (sizeof(T) * count, std::align_val_t(64));
        for (size_t i = 0; i < count; i++)
                new (ptr + i) T();
        return ptr;
}

template <typename T>
inline void free(T *buf) noexcept
{
        ::operator delete [] (buf, std::align_val_t(64));
}

} // namespace nn::memory
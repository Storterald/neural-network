#pragma once

#include "Base.h"

class Vector {
        friend class Matrix;

public:
        constexpr Vector() = default;
        explicit Vector(uint32_t size);
        Vector(uint32_t size, const float *data);
        Vector(const Vector &other);
        Vector(Vector &&other) noexcept;
        ~Vector();

        Vector &operator= (const Vector &other);
        Vector &operator= (Vector &&other) noexcept;

        Vector operator+ (const Vector &other) const;
        Vector operator- (const Vector &other) const;
        Vector operator* (const Vector &other) const;

        void operator+= (const Vector &other);
        void operator-= (const Vector &other);
        void operator*= (const Vector &other);

        Vector operator* (float scalar) const;
        void operator*= (float scalar);

        [[nodiscard]] __host__ __device__ inline uint32_t size() const noexcept
        {
                return m_size;
        }

        [[nodiscard]] __host__ __device__ inline float *data() const noexcept
        {
                return m_data;
        }

        [[nodiscard]] __host__ __device__ inline float at(
                uint32_t i
        ) const {
#ifdef __CUDA_ARCH__
#ifdef DEBUG_MODE_ENABLED
                if (i >= m_size)
                        return NAN;
#endif // DEBUG_MODE_ENABLED
                return m_data[i];
#else
#ifdef DEBUG_MODE_ENABLED
                if (i >= m_size)
                        throw Logger::fatal_error("Vector::at index out of bounds.");
#endif // DEBUG_MODE_ENABLED
                float tmp{};
                CUDA_CHECK_ERROR(cudaMemcpy(&tmp, &m_data[i], sizeof(float), cudaMemcpyDeviceToHost),
                        "Could not copy value from GPU to CPU memory.");
                return tmp;
#endif // __CUDA_ARCH__
        }

        [[nodiscard]] __device__ inline float &operator[] (uint32_t i) const {
#ifdef DEBUG_MODE_ENABLED
                if (i >= m_size)
                        return m_data[-1];
#endif // DEBUG_MODE_ENABLED
                return m_data[i];
        }

private:
        uint32_t m_size { 0 };
        float *m_data { nullptr };

};
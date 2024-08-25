#pragma once

#include "Vector.cuh"

class Matrix {
        friend class Vector;

public:
        constexpr Matrix() = default;
        Matrix(uint32_t width, uint32_t height);
        Matrix(uint32_t width, uint32_t height, const float *data);
        Matrix(const Matrix &other);
        Matrix(Matrix &&other) noexcept;
        ~Matrix();

        Matrix &operator= (const Matrix &other);
        Matrix &operator= (Matrix &&other) noexcept;

        void operator+= (const Matrix &other);
        void operator-= (const Matrix &other);

        // Matrix product visualizer:
        // http://matrixmultiplication.xyz
        Vector operator* (const Vector &vec) const;
        Matrix operator* (float scalar) const;
        void operator*= (float scalar);

        [[nodiscard]] __host__ __device__ inline uint32_t size() const noexcept
        {
                return m_size;
        }

        [[nodiscard]] __host__ __device__ inline uint32_t width() const noexcept
        {
                return m_width;
        }

        [[nodiscard]] __host__ __device__ inline uint32_t height() const noexcept
        {
                return m_height;
        }

        [[nodiscard]] __host__ __device__ inline float *data() const noexcept
        {
                return m_data;
        }

        [[nodiscard]] __host__ __device__ inline float at(
                uint32_t row,
                uint32_t col
        ) const {
#ifdef __CUDA_ARCH__
#ifdef DEBUG_MODE_ENABLED
                if (row >= m_height || col >= m_width)
                        return NAN;
#endif // DEBUG_MODE_ENABLED
                return m_data[row * m_width + col];
#else
#ifdef DEBUG_MODE_ENABLED
                if (row >= m_height || col >= m_width)
                        throw Logger::fatal_error("Matrix::at index out of bounds.");
#endif // DEBUG_MODE_ENABLED
                float tmp{};
                CUDA_CHECK_ERROR(cudaMemcpy(&tmp, &m_data[row * m_width + col], sizeof(float), cudaMemcpyDeviceToHost),
                        "Could not copy value from GPU to CPU memory.");
                return tmp;
#endif // __CUDA_ARCH__
        }

        [[nodiscard]] __device__ inline float *operator[] (
                uint32_t row
        ) {
#ifdef DEBUG_MODE_ENABLED
                if (row >= m_height)
                        return nullptr;
#endif // DEBUG_MODE_ENABLED
                return m_data + row * m_width;
        }

private:
        uint32_t m_width { 0 };
        uint32_t m_height { 0 };
        uint32_t m_size { 0 };
        float *m_data { nullptr };

};
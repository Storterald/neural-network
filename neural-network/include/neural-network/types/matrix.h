#pragma once

#include <initializer_list>
#include <iterator>  // std::data
#include <cstdint>
#include <cstring>   // std::memcpy

#include <neural-network/utils/exceptions.h>
#include <neural-network/types/vector.h>
#include <neural-network/types/buf.h>
#include <neural-network/math/math.h>
#include <neural-network/base.h>

#ifdef DEBUG_MODE_ENABLED
#include <algorithm>  // std::ranges::any_of
#endif // DEBUG_MODE_ENABLED

#ifdef BUILD_CUDA_SUPPORT
#include <neural-network/utils/cuda.h>
#endif // BUILD_CUDA_SUPPORT

namespace nn {

enum memory_layout {
        row_major    = true,
        column_major = false

};

struct indexer {
        uint32_t row;
        uint32_t col;

};

template<typename _type>
class matrix : public buf<_type> {
public:
        using base = buf<_type>;
        using typename base::value_type;
        using typename base::const_value_type;
        using typename base::size_type;
        using typename base::pointer;
        using typename base::const_pointer;
        using typename base::reference;
        using typename base::const_reference;
        using typename base::difference_type;

        inline matrix() = default;
        inline matrix(std::nullptr_t) : matrix() {}

        inline matrix(
                uint32_t width,
                uint32_t height,
                stream_t stream = invalid_stream) :
                        matrix(row_major, width, height, stream) {}
        
        inline matrix(
                memory_layout layout,
                uint32_t      width,
                uint32_t      height,
                stream_t      stream = invalid_stream);
        
        inline matrix(
                std::initializer_list<std::initializer_list<value_type>> values,
                stream_t                                                 stream = invalid_stream) :
                        matrix(values, row_major, stream) {}

        inline matrix(
                std::initializer_list<std::initializer_list<value_type>> values,
                memory_layout                                            layout,
                stream_t                                                 stream = invalid_stream);

        inline matrix(const matrix &other);
        inline matrix &operator= (const matrix &other);

        inline matrix(matrix &&other) noexcept;
        inline matrix &operator= (matrix &&other) noexcept;

        [[nodiscard]] inline bool operator== (const matrix &other) const;

        [[nodiscard]] inline vector<value_type> operator[] (uint32_t idx);
        [[nodiscard]] inline vector<const_value_type> operator[] (uint32_t idx) const;
        [[nodiscard]] inline vector<const_value_type> at(uint32_t idx) const;

#ifndef __CUDACC__  // nvcc does not yet support C++23
        [[nodiscard]] inline reference operator[] (uint32_t row, uint32_t col);
        [[nodiscard]] inline const_reference operator[] (uint32_t row, uint32_t col) const;
#endif // !__CUDACC__

        [[nodiscard]] inline reference operator[] (indexer pos);
        [[nodiscard]] inline const_reference operator[] (indexer pos) const;
        [[nodiscard]] inline value_type at(uint32_t row, uint32_t col) const;

        [[nodiscard]] inline matrix operator+ (const matrix &other) const;
        [[nodiscard]] inline matrix operator- (const matrix &other) const;
        [[nodiscard]] inline matrix operator* (const matrix &other) const;

        inline void operator+= (const matrix &other);
        inline void operator-= (const matrix &other);

        [[nodiscard]] inline matrix operator+ (value_type scalar) const;
        [[nodiscard]] inline matrix operator- (value_type scalar) const;
        [[nodiscard]] inline matrix operator* (value_type scalar) const;
        [[nodiscard]] inline matrix operator/ (value_type scalar) const;

        inline void operator+= (value_type scalar);
        inline void operator-= (value_type scalar);
        inline void operator*= (value_type scalar);
        inline void operator/= (value_type scalar);

        [[nodiscard]] inline matrix<value_type> operator+ (const vector<value_type> &vec) const;
        [[nodiscard]] inline matrix<value_type> operator- (const vector<value_type> &vec) const;
        [[nodiscard]] inline vector<value_type> operator* (const vector<value_type> &vec) const;

        [[nodiscard]] inline size_type width() const noexcept;
        [[nodiscard]] inline size_type height() const noexcept;
        [[nodiscard]] inline memory_layout layout() const noexcept;

        [[nodiscard]] inline matrix transposed_view() const;
        [[nodiscard]] inline matrix transpose() const;

private:
        memory_layout m_layout = row_major;
        uint32_t      m_width  = 0;
        uint32_t      m_height = 0;

        inline matrix(typename base::not_owned_t, const matrix &self);

        [[nodiscard]] inline uint32_t _dim1() const;
        [[nodiscard]] inline uint32_t _dim2() const;

}; // class matrix

template<typename _type>
inline matrix<_type>::matrix(
        memory_layout layout,
        uint32_t      width,
        uint32_t      height,
        stream_t      stream) :
                base(width * height, stream),
                m_layout(layout),
                m_width(width),
                m_height(height) {}

template<typename _type>
inline matrix<_type>::matrix(
        std::initializer_list<std::initializer_list<value_type>> values,
        memory_layout                                            layout,
        stream_t                                                 stream) :
                base((uint32_t)(values.begin()->size() * values.size()), stream),
                m_layout(layout),
                m_width(static_cast<uint32_t>(m_layout ? values.begin()->size() : values.size())),
                m_height(static_cast<uint32_t>(m_layout ? values.size() : values.begin()->size())) {

#ifdef DEBUG_MODE_ENABLED
        if (std::ranges::any_of(values, [this](auto &vs) -> bool { return vs.size() != _dim1(); }))
                throw fatal_error("All initializer lists passed to nn::matrix() must have the same size.");
#endif // DEBUG_MODE_ENABLED

        const size_t size = _dim1() * sizeof(value_type);
        for (uint32_t i = 0; i < _dim2(); ++i) {
                value_type *dst       = this->m_data + i * _dim1();
                const_value_type *src = std::data(std::data(values)[i]);
                if (!this->m_device)
                        std::memcpy(dst, src, size);
#ifdef BUILD_CUDA_SUPPORT
                else
                        cuda::memcpy(dst, src, size, cudaMemcpyHostToDevice, this->m_stream);
#endif // BUILD_CUDA_SUPPORT
        }
}

template<typename _type>
inline matrix<_type>::matrix(const matrix &other) :
        base(other),
        m_layout(other.m_layout),
        m_width(other.m_width),
        m_height(other.m_height) {}

template<typename _type>
inline matrix<_type> &matrix<_type>::operator= (const matrix &other)
{
        base::operator=(other);
        m_layout = other.m_layout;
        m_width  = other.m_width;
        m_height = other.m_height;
        return *this;
}

template<typename _type>
inline matrix<_type>::matrix(matrix &&other) noexcept :
        base(std::move(other)),
        m_layout(other.m_layout),
        m_width(other.m_width),
        m_height(other.m_height) {}

template<typename _type>
inline matrix<_type> &matrix<_type>::operator= (matrix &&other) noexcept
{
        base::operator=(std::move(other));
        m_layout = other.m_layout;
        m_width  = other.m_width;
        m_height = other.m_height;
        return *this;
}

template<typename _type>
inline bool matrix<_type>::operator== (const matrix &other) const
{
        if (this->m_size != other.m_size || this->m_layout != other.m_layout)
                return false;

        bool ans;
        math::compare(this->m_size, *this, other, &ans);
        return ans;
}

template<typename _type>
inline auto matrix<_type>::operator[] (uint32_t idx) -> vector<value_type>
{
#ifdef DEBUG_MODE_ENABLED
        if (idx >= _dim2())
                throw fatal_error("matrix::operator[] access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        return vector<value_type>(base::not_owned, this->m_data + idx * _dim1(), _dim1(), *this);
}

template<typename _type>
inline auto matrix<_type>::operator[] (uint32_t idx) const -> vector<const_value_type>
{
#ifdef DEBUG_MODE_ENABLED
        if (idx >= _dim2())
                throw fatal_error("matrix::operator[] access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        return vector<const_value_type>(base::not_owned, this->m_data + idx * _dim1(), _dim1(), *this);
}

template<typename _type>
inline auto matrix<_type>::at(uint32_t idx) const -> vector<const_value_type>
{
#ifdef DEBUG_MODE_ENABLED
        if (idx >= m_height)
                throw fatal_error("matrix::at access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        return vector<const_value_type>(base::not_owned, this->m_data + idx * _dim1(), _dim1(), *this);
}

#ifndef __CUDACC__  // nvcc does not yet support C++23
template<typename _type>
inline auto matrix<_type>::operator[] (uint32_t row, uint32_t col) -> reference
{
        return this->operator[]({row, col});
}

template<typename _type>
inline auto matrix<_type>::operator[] (uint32_t row, uint32_t col) const -> const_reference
{
        return this->operator[]({row, col});
}
#endif // !__CUDACC__

template<typename _type>
inline auto matrix<_type>::operator[] (indexer pos) -> reference
{
#ifdef DEBUG_MODE_ENABLED
        if (pos.row >= m_height || pos.col >= m_width)
                throw fatal_error("matrix::operator[] access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        const auto offset = static_cast<difference_type>(m_layout == row_major ? pos.row * m_width + pos.col : pos.col * m_height + pos.row);
        return *(this->begin() + offset);
}

template<typename _type>
inline auto matrix<_type>::operator[] (indexer pos) const -> const_reference
{
#ifdef DEBUG_MODE_ENABLED
        if (pos.row >= m_height || pos.col >= m_width)
                throw fatal_error("matrix::operator[] access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        const auto offset = static_cast<difference_type>(m_layout == row_major ? pos.row * m_width + pos.col : pos.col * m_height + pos.row);
        return *(this->begin() + offset);
}

template<typename _type>
inline auto matrix<_type>::at(uint32_t row, uint32_t col) const -> value_type
{
#ifdef DEBUG_MODE_ENABLED
        if (row >= m_height || col >= m_width)
                throw fatal_error("matrix::at access index out of bounds.");
#endif // DEBUG_MODE_ENABLED

        const auto offset = static_cast<difference_type>(m_layout == row_major ? row * m_width + col : col * m_width + row);
        return *(this->begin() + offset);
}

template<typename _type>
inline matrix<_type> matrix<_type>::operator+ (const matrix &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw fatal_error("Matrix dimensions must match for addition.");
#endif // DEBUG_MODE_ENABLED

        matrix<_type> result(m_width, m_height, this->m_stream);
        if (m_layout == other.m_layout)
                math::sum(this->m_size, *this, other, result);
        else
                math::sum(this->m_size, *this, other.transpose(), result); // TODO bad

        return result;
}

template<typename _type>
inline matrix<_type> matrix<_type>::operator- (const matrix &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw fatal_error("Matrix dimensions must match for subtraction.");
#endif // DEBUG_MODE_ENABLED

        matrix<_type> result(m_width, m_height, this->m_stream);
        if (m_layout == other.m_layout)
                math::sub(this->m_size, *this, other, result);
        else
                math::sub(this->m_size, *this, other.transpose(), result); // TODO bad

        return result;
}

template<typename _type>
inline matrix<_type> matrix<_type>::operator* (const matrix &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_height)
                throw fatal_error("Matrix width must match other height for multiplication.");
#endif // DEBUG_MODE_ENABLED

        matrix<_type> result(m_height, other.m_width, this->m_stream);
        if (m_layout == row_major) {
                if (other.m_layout == row_major)
                        math::matmul_rc(m_width, m_height, other.m_width, *this, other.transpose(), result);
                else
                        math::matmul_rc(m_width, m_height, other.m_width, *this, other, result);
        } else {
                if (other.m_layout == row_major)
                        math::matmul_rc(m_width, m_height, other.m_width, this->transpose(), other, result);
                else
                        math::matmul_rc(m_width, m_height, other.m_width, this->transpose(), other.transpose(), result);
        }

        return result;
}

template<typename _type>
inline void matrix<_type>::operator+= (const matrix &other)
{
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw fatal_error("Matrix dimensions must match for addition.");
#endif // DEBUG_MODE_ENABLED

        if (m_layout == other.m_layout)
                math::sum(this->m_size, *this, other, *this);
        else
                math::sum(this->m_size, *this, other.transpose(), *this); // TODO bad
}

template<typename _type>
inline void matrix<_type>::operator-= (const matrix &other)
{
#ifdef DEBUG_MODE_ENABLED
        if (m_width != other.m_width || m_height != other.m_height)
                throw fatal_error("Matrix dimensions must match for subtraction.");
#endif // DEBUG_MODE_ENABLED

        if (m_layout == other.m_layout)
                math::sub(this->m_size, *this, other, *this);
        else
                math::sub(this->m_size, *this, other.transpose(), *this); // TODO bad
}

template<typename _type>
inline matrix<_type> matrix<_type>::operator+ (value_type scalar) const
{
        matrix result(m_width, m_height, this->m_stream);
        math::sum(this->m_size, *this, scalar, result);

        return result;
}

template<typename _type>
inline matrix<_type> matrix<_type>::operator- (value_type scalar) const
{
        matrix result(m_width, m_height, this->m_stream);
        math::sub(this->m_size, *this, scalar, result);

        return result;
}

template<typename _type>
inline matrix<_type> matrix<_type>::operator* (value_type scalar) const
{
        matrix result(m_width, m_height, this->m_stream);
        math::mul(this->m_size, *this, scalar, result);

        return result;
}

template<typename _type>
inline matrix<_type> matrix<_type>::operator/ (value_type scalar) const
{
#ifdef DEBUG_MODE_ENABLED
        if (scalar == 0)
                throw fatal_error("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        matrix result(m_width, m_height, this->m_stream);
        math::div(this->m_size, *this, scalar, result);

        return result;
}

template<typename _type>
inline void matrix<_type>::operator+= (value_type scalar)
{
        math::sum(this->m_size, *this, scalar, *this);
}

template<typename _type>
inline void matrix<_type>::operator-= (value_type scalar)
{
        math::sub(this->m_size, *this, scalar, *this);
}

template<typename _type>
inline void matrix<_type>::operator*= (value_type scalar)
{
        math::mul(this->m_size, *this, scalar, *this);
}

template<typename _type>
inline void matrix<_type>::operator/= (value_type scalar)
{
#ifdef DEBUG_MODE_ENABLED
        if (scalar == 0)
                throw fatal_error("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        math::div(this->m_size, *this, scalar, *this);
}

template<typename _type>
inline auto matrix<_type>::operator+ (const vector<value_type> &vec) const -> matrix<value_type>
{
#ifdef DEBUG_MODE_ENABLED
        if (m_width != vec.size())
                throw fatal_error("Matrix width must match vector size for matrix vector addition.");
#endif // DEBUG_MODE_ENABLED

        matrix<_type> result(m_width, m_height, this->m_stream);
        for (uint32_t i = 0; i < m_height; ++i) {
                vector out = result[i];
                math::sum(m_width, this->operator[](i), vec, out);
        }

        return result;
}

template<typename _type>
inline auto matrix<_type>::operator- (const vector<value_type> &vec) const -> matrix<value_type>
{
#ifdef DEBUG_MODE_ENABLED
        if (m_width != vec.size())
                throw fatal_error("Matrix width must match vector size for matrix vector subtraction.");
#endif // DEBUG_MODE_ENABLED

        matrix<_type> result(m_width, m_height, this->m_stream);
        for (uint32_t i = 0; i < m_height; ++i) {
                vector out = result[i];
                math::sub(m_width, this->operator[](i), vec, out);
        }

        return result;
}

template<typename _type>
inline auto matrix<_type>::operator* (const vector<value_type> &vec) const -> vector<value_type>
{
#ifdef DEBUG_MODE_ENABLED
        if (m_width != vec.size())
                throw fatal_error("Matrix width and vector size must match.");
#endif // DEBUG_MODE_ENABLED

        vector<_type> result(m_height, this->m_stream);
        if (m_layout == row_major)
                math::matvec_r(m_width, m_height, *this, vec, result);
        else
                math::matvec_c(m_width, m_height, *this, vec, result);

        return result;
}

template<typename _type>
inline auto matrix<_type>::width() const noexcept -> size_type
{
        return m_width;
}

template<typename _type>
inline auto matrix<_type>::height() const noexcept -> size_type
{
        return m_height;
}

template<typename _type>
inline memory_layout matrix<_type>::layout() const noexcept
{
        return m_layout;
}

template<typename _type>
inline matrix<_type> matrix<_type>::transposed_view() const
{
        matrix shallow   = matrix(base::not_owned, *this);
        shallow.m_layout = static_cast<memory_layout>(!m_layout);
        shallow.m_width  = m_height;
        shallow.m_height = m_width;
        return shallow;
}

template<typename _type>
inline matrix<_type> matrix<_type>::transpose() const
{
        matrix result(m_layout, m_width, m_height, this->m_stream);
        math::transpose(m_width, m_height, *this, result);
        return result;
}

template<typename _type>
matrix<_type>::matrix(typename base::not_owned_t, const matrix &self) :
        buf<_type>(base::not_owned, self),
        m_layout(self.m_layout),
        m_width(self.width()),
        m_height(self.height()) {}

template<typename _type>
inline uint32_t matrix<_type>::_dim1() const {
        if (m_layout == row_major)
                return m_width;
        return m_height;
}

template<typename _type>
inline uint32_t matrix<_type>::_dim2() const {
        if (m_layout == row_major)
                return m_height;
        return m_width;
}

} // namespace nn

template<nn::floating_type _type>
[[nodiscard]] inline nn::matrix<_type> operator+ (_type scalar, const nn::matrix<_type> &mat) {
        return mat + scalar;
}

template<nn::floating_type _type>
[[nodiscard]] inline nn::matrix<_type> operator- (_type scalar, const nn::matrix<_type> &mat) {
        return mat - scalar;
}

template<nn::floating_type _type>
[[nodiscard]] inline nn::matrix<_type> operator* (_type scalar, const nn::matrix<_type> &mat) {
        return mat * scalar;
}

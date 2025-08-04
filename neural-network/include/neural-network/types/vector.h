#pragma once

#include <initializer_list>
#include <iterator>  // std::distance
#include <cstdint>
#include <memory>  // std::to_address
#include <ranges>

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

template<typename _type>
class matrix;

template<typename _type>
class vector : public buf<std::remove_const_t<_type>> {
public:
        using base = buf<std::remove_const_t<_type>>;
        using typename base::value_type;
        using typename base::const_value_type;
        using typename base::size_type;
        using typename base::pointer;
        using typename base::const_pointer;
        using typename base::reference;
        using typename base::const_reference;
        using typename base::difference_type;

        inline vector() = default;
        inline vector(std::nullptr_t) : vector() {}

        inline vector(const vector &other);
        inline vector &operator= (const vector &other);

        inline vector(vector &&other) noexcept;
        inline vector &operator= (vector &&other) noexcept;

        inline explicit vector(
                uint32_t size,
                stream_t stream = invalid_stream);

        template<std::ranges::range values_t>
        requires std::is_same_v<std::ranges::range_value_t<values_t>, value_type>
            && (!std::is_same_v<std::remove_cvref_t<values_t>, vector<value_type>>)
        inline vector(
                values_t &&values,
                stream_t stream = invalid_stream) :
                        vector(std::ranges::begin(values), std::ranges::end(values), stream) {}

        inline vector(
                std::initializer_list<value_type> values,
                stream_t                          stream = invalid_stream) :
                        vector(values.begin(), values.end(), stream) {}

        template<std::input_iterator beg_t, std::sentinel_for<beg_t> end_t>
        inline vector(
                beg_t    begin,
                end_t    end,
                stream_t stream = invalid_stream);

        [[nodiscard]] inline bool operator== (const vector &other) const;

        [[nodiscard]] inline reference operator[] (uint32_t i);
        [[nodiscard]] inline const_reference operator[] (uint32_t i) const;
        [[nodiscard]] inline value_type at(uint32_t i) const;

        [[nodiscard]] inline vector min  (value_type min) const;
        [[nodiscard]] inline vector max  (value_type max) const;
        [[nodiscard]] inline vector clamp(value_type min, value_type max) const;

        [[nodiscard]] inline vector min  (const vector &other) const;
        [[nodiscard]] inline vector max  (const vector &other) const;
        [[nodiscard]] inline vector clamp(const vector &min, const vector &max) const;

        [[nodiscard]] inline vector operator+ (const vector &other) const;
        [[nodiscard]] inline vector operator- (const vector &other) const;
        [[nodiscard]] inline vector operator* (const vector &other) const;
        [[nodiscard]] inline vector operator/ (const vector &other) const;

        inline void operator+= (const vector &other);
        inline void operator-= (const vector &other);
        inline void operator*= (const vector &other);
        inline void operator/= (const vector &other);

        [[nodiscard]] inline vector operator+ (value_type scalar) const;
        [[nodiscard]] inline vector operator- (value_type scalar) const;
        [[nodiscard]] inline vector operator* (value_type scalar) const;
        [[nodiscard]] inline vector operator/ (value_type scalar) const;

        inline void operator+= (value_type scalar);
        inline void operator-= (value_type scalar);
        inline void operator*= (value_type scalar);
        inline void operator/= (value_type scalar);

private:
        inline vector(typename base::not_owned_t, value_type *ptr, size_type size, const buf<value_type> &self);

        friend class matrix<value_type>;

}; // class vector

template<typename _type>
inline vector<_type>::vector(
        uint32_t size,
        stream_t stream) :
                base(size, stream) {}

template<typename _type>
template<std::input_iterator beg_t, std::sentinel_for<beg_t> end_t>
inline vector<_type>::vector(
        beg_t    begin,
        end_t    end,
        stream_t stream) :
                base(static_cast<uint32_t>(std::distance(begin, end)), stream) {

        const value_type *data = std::to_address(begin);
        if (!this->m_device)
                std::memcpy(this->m_data, data, this->m_size * sizeof(value_type));
#ifdef BUILD_CUDA_SUPPORT
        else
                cuda::memcpy(this->m_data, data, this->m_size * sizeof(value_type),
                        cudaMemcpyHostToDevice, this->m_stream);
#endif // BUILD_CUDA_SUPPORT
}

template<typename _type>
inline vector<_type>::vector(const vector &other) :
        base(other) {}

template<typename _type>
inline vector<_type> &vector<_type>::operator= (const vector &other)
{
        base::operator=(other);
        return *this;
}

template<typename _type>
inline vector<_type>::vector(vector &&other) noexcept :
        base(std::move(other)) {}

template<typename _type>
inline vector<_type> &vector<_type>::operator= (vector &&other) noexcept
{
        base::operator=(std::move(other));
        return *this;
}

template<typename _type>
inline bool vector<_type>::operator== (const vector &other) const
{
        if (this->m_size != other.m_size)
                return false;

        bool ans;
        math::compare(this->m_size, *this, other, &ans);
        return ans;
}

template<typename _type>
inline auto vector<_type>::operator[] (uint32_t i) -> reference
{
#ifdef DEBUG_MODE_ENABLED
        if (i >= this->m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        return *(this->begin() + i);
}

template<typename _type>
inline auto vector<_type>::operator[] (uint32_t i) const -> const_reference
{
#ifdef DEBUG_MODE_ENABLED
        if (i >= this->m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        return *(this->begin() + i);
}

template<typename _type>
inline auto vector<_type>::at(uint32_t i) const -> value_type
{
#ifdef DEBUG_MODE_ENABLED
        if (i >= this->m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        return *(this->begin() + i);
}

template<typename _type>
inline vector<_type> vector<_type>::operator+ (const vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (this->m_size != other.m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        vector result(this->m_size, this->m_stream);
        math::sum(this->m_size, *this, other, result);

        return result;
}

template<typename _type>
inline vector<_type> vector<_type>::operator- (const vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (this->m_size != other.m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        vector result(this->m_size, this->m_stream);
        math::sub(this->m_size, *this, other, result);

        return result;
}

template<typename _type>
inline vector<_type> vector<_type>::operator* (const vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (this->m_size != other.m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        vector result(this->m_size, this->m_stream);
        math::mul(this->m_size, *this, other, result);

        return result;
}

template<typename _type>
inline vector<_type> vector<_type>::operator/ (const vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (this->m_size != other.m_size)
                throw fatal_error("Vector sizes must match.");
        if (std::ranges::any_of(other, [](float v) -> bool { return v == 0.0f; }))
                throw fatal_error("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        vector result(this->m_size, this->m_stream);
        math::div(this->m_size, *this, other, result);

        return result;
}

template<typename _type>
inline void vector<_type>::operator+= (const vector &other)
{
#ifdef DEBUG_MODE_ENABLED
        if (this->m_size != other.m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        math::sum(this->m_size, *this, other, *this);
}

template<typename _type>
inline void vector<_type>::operator-= (const vector &other)
{
#ifdef DEBUG_MODE_ENABLED
        if (this->m_size != other.m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        math::sub(this->m_size, *this, other, *this);
}

template<typename _type>
inline void vector<_type>::operator*= (const vector &other)
{
#ifdef DEBUG_MODE_ENABLED
        if (this->m_size != other.m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        math::mul(this->m_size, *this, other, *this);
}

template<typename _type>
inline void vector<_type>::operator/= (const vector &other)
{
#ifdef DEBUG_MODE_ENABLED
        if (this->m_size != other.m_size)
                throw fatal_error("Vector sizes must match.");
        if (std::ranges::any_of(other, [](float v) -> bool { return v == 0.0f; }))
                throw fatal_error("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        math::div(this->m_size, *this, other, *this);
}

template<typename _type>
inline vector<_type> vector<_type>::operator+ (value_type scalar) const
{
        vector result(this->m_size, this->m_stream);
        math::sum(this->m_size, *this, scalar, result);

        return result;
}

template<typename _type>
inline vector<_type> vector<_type>::operator- (value_type scalar) const
{
        vector result(this->m_size, this->m_stream);
        math::sub(this->m_size, *this, scalar, result);

        return result;
}

template<typename _type>
inline vector<_type> vector<_type>::operator* (value_type scalar) const
{
        vector result(this->m_size, this->m_stream);
        math::mul(this->m_size, *this, scalar, result);

        return result;
}

template<typename _type>
inline vector<_type> vector<_type>::operator/ (value_type scalar) const
{
#ifdef DEBUG_MODE_ENABLED
        if (scalar == 0.0f)
                throw fatal_error("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        vector result(this->m_size, this->m_stream);
        math::div(this->m_size, *this, scalar, result);

        return result;
}

template<typename _type>
inline void vector<_type>::operator+= (value_type scalar)
{
        math::sum(this->m_size, *this, scalar, *this);
}

template<typename _type>
inline void vector<_type>::operator-= (value_type scalar)
{
        math::sub(this->m_size, *this, scalar, *this);
}

template<typename _type>
inline void vector<_type>::operator*= (value_type scalar)
{
        math::mul(this->m_size, *this, scalar, *this);
}

template<typename _type>
inline void vector<_type>::operator/= (value_type scalar)
{
#ifdef DEBUG_MODE_ENABLED
        if (scalar == 0)
                throw fatal_error("Cannot divide by 0.");
#endif // DEBUG_MODE_ENABLED

        math::div(this->m_size, *this, scalar, *this);
}

template<typename _type>
inline vector<_type> vector<_type>::min(value_type min) const
{
        vector result(this->m_size, this->m_stream);
        math::min(this->m_size, *this, min, result);

        return result;
}

template<typename _type>
inline vector<_type> vector<_type>::max(value_type max) const
{
        vector result(this->m_size, this->m_stream);
        math::max(this->m_size, *this, max, result);

        return result;
}

template<typename _type>
inline vector<_type> vector<_type>::clamp(value_type min, value_type max) const
{
        vector result(this->m_size, this->m_stream);
        math::clamp(this->m_size, *this, min, max, result);

        return result;
}

template<typename _type>
inline vector<_type> vector<_type>::min(const vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (this->m_size != other.m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        vector result(this->m_size, this->m_stream);
        math::min(this->m_size, *this, other, result);

        return result;
}

template<typename _type>
inline vector<_type> vector<_type>::max(const vector &other) const
{
#ifdef DEBUG_MODE_ENABLED
        if (this->m_size != other.m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        vector result(this->m_size, this->m_stream);
        math::max(this->m_size, *this, other, result);

        return result;
}

template<typename _type>
inline vector<_type> vector<_type>::clamp(const vector &min, const vector &max) const
{
#ifdef DEBUG_MODE_ENABLED
        if (this->m_size != min.m_size || this->m_size != max.m_size)
                throw fatal_error("Vector sizes must match.");
#endif // DEBUG_MODE_ENABLED

        vector result(this->m_size, this->m_stream);
        math::clamp(this->m_size, *this, min, max, result);

        return result;
}

template<typename _type>
inline vector<_type>::vector(typename base::not_owned_t, value_type *data, size_type size, const buf<value_type> &self) :
        base(base::not_owned, self) {

        this->m_data = data;
        this->m_size = size;
}

} // namespace nn

template<nn::floating_type _type>
[[nodiscard]] inline nn::vector<_type> operator+ (float scalar, const nn::vector<_type> &vec) {
        return vec + scalar;
}

template<nn::floating_type _type>
[[nodiscard]] inline nn::vector<_type> operator- (float scalar, const nn::vector<_type> &vec) {
        return vec - scalar;
}

template<nn::floating_type _type>
[[nodiscard]] inline nn::vector<_type> operator* (float scalar, const nn::vector<_type> &vec) {
        return vec * scalar;
}

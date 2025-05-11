#pragma once

#include <type_traits>
#include <cstddef> // ptrdiff_t

#include <neural-network/utils/logger.h>

namespace nn {

template<typename>
class ptr;

template<typename>
class ref;

template<typename>
class span;

/**
 * \brief Pointer wrapper around a variable of type \code T\endcode, this class
 * will likely be completely optimized and treated as \code T *\endcode since
 * this class serves a purpose only when compiling with \code BUILD_CUDA_SUPPORT\endcode.
 *
 * \tparam T The pointed type.
 */
template<typename T>
class ptr {
        friend class ref<T>;

public:
        static_assert(std::is_trivially_copyable_v<T>,
                "Ptr<T> requires T to be trivially copiable. "
                "https://en.cppreference.com/w/cpp/named_req/TriviallyCopyable");

        using value_type = T;
        using const_value_type = std::add_const_t<T>;
        using difference_type = ptrdiff_t;

        inline ptr() noexcept : m_pointer(nullptr) {}

        ptr(T *ptr, bool, void * = 0) noexcept : m_pointer(ptr) {}

        template<typename other_type>
        [[nodiscard]] constexpr bool operator== (const ptr<other_type> &other) const noexcept
        {
                return m_pointer == other.get();
        }

        template<typename other_type>
        [[nodiscard]] constexpr bool operator!= (const ptr<other_type> &other) const noexcept
        {
                return !this->operator==(other);
        }

        [[nodiscard]] constexpr bool operator== (const void *other) const noexcept
        {
                return m_pointer == other;
        }

        [[nodiscard]] constexpr bool operator!= (const void *other) const noexcept
        {
                return !this->operator==(other);
        }

        [[nodiscard]] constexpr ref<value_type> operator* () const
        {
#ifdef DEBUG_MODE_ENABLED
                if (!m_pointer)
                        throw LOGGER_EX("Cannot dereference null pointer.");
#endif // DEBUG_MODE_ENABLED

                return { m_pointer, false };
        }

        [[nodiscard]] constexpr ref<value_type> operator[] (uint32_t i) const
        {
#ifdef DEBUG_MODE_ENABLED
                if (!m_pointer)
                        throw LOGGER_EX("Cannot index null pointer.");
#endif // DEBUG_MODE_ENABLED

                return { m_pointer + i, false };
        }

        [[nodiscard]] constexpr ptr operator+ (difference_type offset) const noexcept
        {
                return { m_pointer + offset, false };
        }

        [[nodiscard]] constexpr ptr operator- (difference_type offset) const noexcept
        {
                return { m_pointer - offset, false };
        }

        [[nodiscard]] constexpr difference_type operator- (const ptr &other) const noexcept
        {
                return m_pointer - other.m_pointer;
        }

        constexpr ptr &operator+= (uint32_t offset) noexcept
        {
                m_pointer += offset;
                return *this;
        }

        constexpr ptr &operator-= (uint32_t offset) noexcept
        {
                m_pointer -= offset;
                return *this;
        }

        [[nodiscard]] constexpr ptr operator++ (int) noexcept
        {
                ++m_pointer;
                return { m_pointer + 1, false };
        }

        [[nodiscard]] constexpr ptr operator-- (int) noexcept
        {
                --m_pointer;
                return { m_pointer - 1, false };
        }

        constexpr ptr &operator++ () noexcept
        {
                ++m_pointer;
                return *this;
        }

        constexpr ptr &operator-- () noexcept
        {
                --m_pointer;
                return *this;
        }

        [[nodiscard]] constexpr value_type *get() noexcept
        {
                return m_pointer;
        }

        [[nodiscard]] constexpr const_value_type *get() const noexcept
        {
                return m_pointer;
        }

        [[nodiscard]] constexpr bool is_device() const noexcept
        {
                return false;
        }

        [[nodiscard]] span<value_type> make_span(uint32_t size, bool, bool = false)
        {
                return { size, false, m_pointer, false, false };
        }

        [[nodiscard]] span<const_value_type> make_span(uint32_t size, bool, bool = false) const
        {
                return { size, false, (const_value_type *)m_pointer, false, false };
        }

protected:
        value_type        *m_pointer;

}; // class ptr

/**
 * \brief Reference wrapper around a variable of type \code T\endcode. Like
 * \code Ptr<T>\endcode, this class will likely be optimized as \code T &\endcode.
 *
 * \tparam T The referred variable type.
 */
template<typename T>
class ref {
        using arg_type = std::conditional_t<sizeof(T) <= 8, T, const T &>;

public:
        using value_type = T;
        using const_value_type = std::add_const_t<T>;

        ref(T *pValue, bool, void * = 0) : m_ptr(pValue, false) {
                if (!pValue)
                        throw LOGGER_EX("Cannot create null reference.");
        }

        ref(const ref &other) = delete;
        ref &operator= (const ref &other) noexcept
        {
                if (this != std::addressof(other))
                        *m_ptr.m_pointer = *other.m_ptr.m_pointer;

                return *this;
        }

        ref(ref &&other) noexcept : m_ptr(other.m_ptr) {}
        ref &operator= (ref &&other) noexcept
        {
                if (this != std::addressof(other))
                        *m_ptr.m_pointer = *other.m_ptr.m_pointer;

                return *this;
        }

        ref &operator= (arg_type value) noexcept requires (!std::is_const_v<value_type>)
        {
                *m_ptr.m_pointer = value;
                return *this;
        }

        [[nodiscard]] constexpr bool operator== (const ref &other) const noexcept
        {
                return *m_ptr.m_pointer == *other.m_ptr.m_pointer;
        }

        [[nodiscard]] constexpr bool operator!= (const ref &other) const noexcept
        {
                return !this->operator==(other);
        }

        [[nodiscard]] constexpr ptr<value_type> operator& () const noexcept
        {
                return m_ptr;
        }

        [[nodiscard]] constexpr operator value_type() const noexcept
        {
                return *m_ptr.m_pointer;
        }

protected:
        ptr<value_type>        m_ptr;

}; // class ref

/**
 * \brief A Non-owning span around some memory. This class can be treated as a
 * \code T *\endcode since the memory wrapped around a span will always be in
 * the host memory.
 *
 * \tparam T The type of the span pointer
 */
template<typename T>
class span {
        using raw_type = std::remove_const_t<T>;

public:
        using value_type = T;
        using const_value_type = std::add_const_t<T>;

        span(uint32_t size, bool, T *src, bool, bool = false, void * = 0) noexcept :
                m_ptr(src), m_size(size) {}

        [[nodiscard]] constexpr operator value_type *() noexcept
        {
                return m_ptr;
        }

        [[nodiscard]] constexpr operator const_value_type *() const noexcept
        {
                return m_ptr;
        }

        [[nodiscard]] inline ref<value_type> operator[] (uint32_t i)
        {
                return { m_ptr + i, false };
        }

        [[nodiscard]] inline ref<const_value_type> operator[] (uint32_t i) const
        {
                return { m_ptr + i, false };
        }

        [[nodiscard]] inline ref<value_type> begin()
        {
                return { m_ptr, false };
        }

        [[nodiscard]] inline ref<const_value_type> begin() const
        {
                return { m_ptr, false };
        }

        [[nodiscard]] inline ref<value_type> end()
        {
                return { m_ptr + m_size, false };
        }

        [[nodiscard]] inline ref<const_value_type> end() const
        {
                return { m_ptr + m_size, false };
        }

        [[nodiscard]] constexpr uint32_t size() const noexcept
        {
                return m_size;
        }

        [[nodiscard]] constexpr bool is_owning() const noexcept
        {
                return false;
        }

        constexpr void update() const noexcept {}

private:
        value_type        *m_ptr;
        uint32_t          m_size;

}; // class span

} // namespace nn

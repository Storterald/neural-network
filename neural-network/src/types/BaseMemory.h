#pragma once

#include "../Base.h"

template<typename>
class Ptr;

template<typename>
class Ref;

template<typename>
class Span;

/**
 * \brief Pointer wrapper around a variable of type \code T\endcode, this class
 * will likely be completely optimized and treated as \code T *\endcode since
 * this class serves a purpose only when compiling with \code BUILD_CUDA_SUPPORT\endcode.
 *
 * \tparam T The pointed type.
 */
template<typename T>
class Ptr {
        friend class Ref<T>;

public:
        using value_type = T;
        using difference_type = ptrdiff_t;

        inline Ptr() : m_pointer(nullptr) {}

        Ptr(T *ptr, bool) : m_pointer(ptr) {}

        [[nodiscard]] constexpr bool operator== (const Ptr &other) const noexcept
        {
                return m_pointer == other.m_pointer;
        }

        [[nodiscard]] constexpr bool operator!= (const Ptr &other) const noexcept
        {
                return !this->operator==(other);
        }

        [[nodiscard]] constexpr bool operator== (const T *other) const noexcept
        {
                return m_pointer == other;
        }

        [[nodiscard]] constexpr bool operator!= (const T *other) const noexcept
        {
                return !this->operator==(other);
        }

        [[nodiscard]] constexpr Ref<T> operator* () const
        {
#ifdef DEBUG_MODE_ENABLED
                if (!m_pointer)
                        throw LOGGER_EX("Cannot dereference null pointer.");
#endif // DEBUG_MODE_ENABLED

                return Ref<T>(m_pointer, false);
        }

        [[nodiscard]] constexpr Ref<T> operator[] (uint32_t i) const
        {
#ifdef DEBUG_MODE_ENABLED
                if (!m_pointer)
                        throw LOGGER_EX("Cannot index null pointer.");
#endif // DEBUG_MODE_ENABLED

                return { m_pointer + i, false };
        }

        [[nodiscard]] constexpr Ptr operator+ (uint32_t offset)
        {
                return { m_pointer + offset, false };
        }

        [[nodiscard]] constexpr Ptr operator- (uint32_t offset)
        {
                return { m_pointer - offset, false };
        }

        constexpr Ptr &operator+= (uint32_t offset)
        {
                m_pointer += offset;
                return *this;
        }

        constexpr Ptr &operator-= (uint32_t offset)
        {
                m_pointer -= offset;
                return *this;
        }

        [[nodiscard]] constexpr Ptr operator++ (int)
        {
                return { m_pointer + 1, false };
        }

        [[nodiscard]] constexpr Ptr operator-- (int)
        {
                return { m_pointer - 1, false };
        }

        constexpr Ptr &operator++ ()
        {
                ++m_pointer;
                return *this;
        }

        constexpr Ptr &operator-- ()
        {
                --m_pointer;
                return *this;
        }

        [[nodiscard]] constexpr T *get() const noexcept
        {
                return m_pointer;
        }

        [[nodiscard]] constexpr bool is_device() const noexcept
        {
                return false;
        }

        [[nodiscard]] Span<T> span(uint32_t size, bool, bool = false) const
        {
                return { size, false, m_pointer, false, false };
        }

protected:
        T        *m_pointer;

}; // class Ptr

/**
 * \brief Reference wrapper around a variable of type \code T\endcode. Like
 * \code Ptr<T>\endcode, this class will likely be optimized as \code T &\endcode.
 *
 * \tparam T The referred variable type.
 */
template<typename T>
class Ref {
        using ArgType = std::conditional_t<sizeof(T) <= 8, T, const T &>;

public:
        using value_type = T;

        Ref(T *pValue, bool) : m_ptr(pValue, false) {
                if (!pValue)
                        throw LOGGER_EX("Cannot create null reference.");
        }

        Ref(const Ref &other) = delete;
        Ref &operator= (const Ref &other)
        {
                if (this != std::addressof(other))
                        *m_ptr.m_pointer = *other.m_ptr.m_pointer;

                return *this;
        }

        Ref(Ref &&other) noexcept : m_ptr(other.m_ptr) {}
        Ref &operator= (Ref &&other) noexcept
        {
                if (this != std::addressof(other))
                        *m_ptr.m_pointer = *other.m_ptr.m_pointer;

                return *this;
        }

        Ref &operator= (ArgType value)
        {
                *m_ptr.m_pointer = value;
                return *this;
        }

        [[nodiscard]] constexpr bool operator== (const Ref &other) const noexcept
        {
                return *m_ptr.m_pointer == *other.m_ptr.m_pointer;
        }

        [[nodiscard]] constexpr bool operator!= (const Ref &other) const noexcept
        {
                return !this->operator==(other);
        }

        [[nodiscard]] constexpr Ptr<T> &operator& () noexcept
        {
                return m_ptr;
        }

        [[nodiscard]] constexpr const Ptr<T> &operator& () const noexcept
        {
                return m_ptr;
        }

        [[nodiscard]] constexpr operator T() const noexcept
        {
                return *m_ptr.m_pointer;
        }

protected:
        Ptr<T>        m_ptr;

}; // class Ref

/**
 * \brief A Non-owning span around some memory. This class can be treated as a
 * \code T *\endcode since the memory wrapped around a span will always be in
 * the host memory.
 *
 * \tparam T The type of the span pointer
 */
template<typename T>
class Span {
public:
        using value_type = T;

        Span(uint32_t size, bool, T *src, bool, bool = false) : m_ptr(src), m_size(size) {}

        [[nodiscard]] constexpr operator T *() noexcept
        {
                return m_ptr;
        }

        [[nodiscard]] constexpr operator const T *() const noexcept
        {
                return m_ptr;
        }

        [[nodiscard]] inline Ref<T> operator[] (uint32_t i)
        {
                return { m_ptr + i, false };
        }

        [[nodiscard]] constexpr uint32_t size() const noexcept
        {
                return m_size;
        }

        [[nodiscard]] constexpr bool is_owning() const noexcept
        {
                return false;
        }

        constexpr void update() {}

private:
        T                            *m_ptr;
        uint32_t                     m_size;

}; // class Span

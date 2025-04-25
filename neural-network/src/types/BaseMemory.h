#pragma once

#include "../Base.h"

template<typename>
class Ptr;

template<typename>
class Ref;

template<typename>
class Span;

template<typename T>
class Ptr {
        friend class Ref<T>;

public:
        Ptr(T *ptr, bool) : m_pointer(ptr) {}

        [[nodiscard]] constexpr bool operator== (const Ptr<T> &other) const noexcept
        {
                return m_pointer == other.m_pointer;
        }

        [[nodiscard]] constexpr bool operator!= (const Ptr<T> &other) const noexcept
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

        [[nodiscard]] constexpr Ptr<T> operator+ (uint32_t offset)
        {
                return { m_pointer + offset, false };
        }

        [[nodiscard]] constexpr Ptr<T> operator- (uint32_t offset)
        {
                return { m_pointer - offset, false };
        }

        constexpr Ptr<T> &operator+= (uint32_t offset)
        {
                m_pointer += offset;
                return *this;
        }

        constexpr Ptr<T> &operator-= (uint32_t offset)
        {
                m_pointer -= offset;
                return *this;
        }

        [[nodiscard]] constexpr Ptr<T> operator++ (int)
        {
                return { m_pointer + 1, false };
        }

        [[nodiscard]] constexpr Ptr<T> operator-- (int)
        {
                return { m_pointer - 1, false };
        }

        constexpr Ptr<T> &operator++ ()
        {
                ++m_pointer;
                return *this;
        }

        constexpr Ptr<T> &operator-- ()
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

        [[nodiscard]] Span<T> span(uint32_t size, bool, bool updateOnDestruction = false) const
        {
                return { size, false, m_pointer, false, updateOnDestruction };
        }

protected:
        T        *m_pointer;

}; // class Ptr

/**
 * @brief Reference wrapper around a type T which may be on the host memory or on the
 * device memory.
 *
 * Uses <code>Ptr\<T\></code> internally.
 *
 * @tparam T The referenced type.
 */
template<typename T>
class Ref {
        using ArgType = std::conditional_t<(sizeof(T) <= 8), T, const T &>;

public:
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

        Ref<T> &operator= (ArgType value)
        {
                *m_ptr.m_pointer = value;
                return *this;
        }

        [[nodiscard]] constexpr bool operator== (const Ref<T> &other) const noexcept
        {
                return *m_ptr.m_pointer == *other.m_ptr.m_pointer;
        }

        [[nodiscard]] constexpr bool operator!= (const Ref<T> &other) const noexcept
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
 * @brief A owning or non-owning span around some memory that may be on the host
 * or on the device.
 *
 * Span::update() should be called if any modification is done to the pointed data.
 *
 * @tparam T The type of the span pointer
 */
template<typename T>
class Span {
public:
        Span(uint32_t size, bool, T *src, bool, bool = false) : m_ptr(src), m_size(size) {}

        [[nodiscard]] constexpr operator T *() const noexcept
        {
                return m_ptr;
        }

        [[nodiscard]] inline Ref<T> operator[] (uint32_t i) const
        {
                return { m_ptr + i, m_device };
        }

        [[nodiscard]] constexpr uint32_t size() const noexcept
        {
                return m_size;
        }

        [[nodiscard]] constexpr bool is_owning() const noexcept
        {
                return m_owning;
        }

        void update() {}

private:
        T                            *m_ptr;
        uint32_t                     m_size;
        static constexpr bool        m_device = false;
        static constexpr bool        m_owning = false;

}; // class Span

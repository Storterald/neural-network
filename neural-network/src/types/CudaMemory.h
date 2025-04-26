#pragma once

#include "../Base.h"

template<typename>
class Ptr;

template<typename>
class Ref;

template<typename>
class Span;

/**
 * \brief Pointer wrapper around a variable of type \code T\endcode which may be
 * on the host memory or on the device memory.
 *
 * \tparam T The pointed type.
 */
template<typename T>
class Ptr {
        friend class Ref<T>;

public:
        using value_type = T;
        using difference_type = ptrdiff_t;

        inline Ptr() : m_pointer(nullptr), m_value(), m_device(false) {}

        Ptr(T *ptr, bool device) : m_pointer(ptr), m_device(device)
        {
                _update_value();
        }

        Ptr(const Ptr &other)
        {
                m_pointer = other.m_pointer;
                m_value   = other.m_value;
                m_device  = other.m_device;
        }

        Ptr &operator= (const Ptr &other)
        {
                if (this == &other)
                        return *this;

                m_pointer = other.m_pointer;
                m_value   = other.m_value;
                m_device  = other.m_device;

                return *this;
        }

        Ptr(Ptr &&other) noexcept
        {
                m_pointer = other.m_pointer;
                m_value   = other.m_value;
                m_device  = other.m_device;

                other.m_pointer = nullptr;
        }

        Ptr &operator= (Ptr &&other) noexcept
        {
                if (this == &other)
                        return *this;

                m_pointer = other.m_pointer;
                m_value   = other.m_value;
                m_device  = other.m_device;

                other.m_pointer = nullptr;

                return *this;
        }

        [[nodiscard]] constexpr bool operator== (const Ptr &other) const noexcept
        {
                return m_pointer == other.m_pointer && m_device == other.m_device;
        }

        [[nodiscard]] constexpr bool operator!= (const Ptr &other) const noexcept
        {
                return !this->operator==(other);
        }

        /// Warning: Returns true even if location is different
        [[nodiscard]] constexpr bool operator== (const T *other) const noexcept
        {
                return m_pointer == other;
        }

        /// Warning: Returns false even if location is different
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

                return Ref<T>(m_pointer, m_device);
        }

        [[nodiscard]] constexpr Ref<T> operator[] (uint32_t i) const
        {
#ifdef DEBUG_MODE_ENABLED
                if (!m_pointer)
                        throw LOGGER_EX("Cannot index null pointer.");
#endif // DEBUG_MODE_ENABLED

                return { m_pointer + i, m_device };
        }

        [[nodiscard]] constexpr Ptr operator+ (uint32_t offset)
        {
                return { m_pointer + offset, m_device };
        }

        [[nodiscard]] constexpr Ptr operator- (uint32_t offset)
        {
                return { m_pointer - offset, m_device };
        }

        constexpr Ptr &operator+= (uint32_t offset)
        {
                m_pointer += offset;
                _update_value();
                return *this;
        }

        constexpr Ptr &operator-= (uint32_t offset)
        {
                m_pointer -= offset;
                _update_value();
                return *this;
        }

        [[nodiscard]] constexpr Ptr operator++ (int)
        {
                return { m_pointer + 1, m_device };
        }

        [[nodiscard]] constexpr Ptr operator-- (int)
        {
                return { m_pointer - 1, m_device };
        }

        constexpr Ptr &operator++ ()
        {
                ++m_pointer;
                _update_value();
                return *this;
        }

        constexpr Ptr<T> &operator-- ()
        {
                --m_pointer;
                _update_value();
                return *this;
        }

        [[nodiscard]] constexpr T *get() const noexcept
        {
                return m_pointer;
        }

        [[nodiscard]] constexpr bool is_device() const noexcept
        {
                return m_device;
        }

        [[nodiscard]] Span<T> span(uint32_t size, bool device, bool updateOnDestruction = false) const
        {
                return { size, device, m_pointer, m_device, updateOnDestruction };
        }

protected:
        T           *m_pointer;
        T           m_value;
        bool        m_device;

        void _update_value()
        {
                if (!m_pointer)
                        return;

                if (!m_device)
                        m_value = *m_pointer;
                else
                        CUDA_CHECK_ERROR(cudaMemcpy(&m_value, m_pointer, sizeof(T),
                                cudaMemcpyDeviceToHost), "Failed to copy data from the GPU.");
        }

}; // class Ptr

/**
 * \brief Reference wrapper around a variable of type \code T\endcode which may
 * be on the host memory or on the device memory.
 *
 * Uses \code Ptr<T>\endcode internally.
 *
 * \tparam T The referred type.
 */
template<typename T>
class Ref {
        using ArgType = std::conditional_t<sizeof(T) <= 8, T, const T &>;

public:
        using value_type = T;

        Ref(T *pValue, bool device) : m_ptr(pValue, device) {
                if (!pValue)
                        throw LOGGER_EX("Cannot create null reference.");
        }

        Ref(const Ref &other) = delete;
        Ref &operator= (const Ref &other)
        {
                if (this != std::addressof(other))
                        _update_value(other.m_ptr.m_value);

                return *this;
        }

        Ref(Ref &&other) noexcept : m_ptr(other.m_ptr) {}
        Ref &operator= (Ref &&other) noexcept
        {
                if (this != std::addressof(other))
                        _update_value(other.m_ptr.m_value);

                return *this;
        }

        Ref &operator= (ArgType value)
        {
                _update_value(value);
                return *this;
        }

        [[nodiscard]] constexpr bool operator== (const Ref &other) const noexcept
        {
                return m_ptr.m_value == other.m_ptr.m_value;
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
                return m_ptr.m_value;
        }

protected:
        Ptr<T>        m_ptr;

        void _update_value(ArgType value)
        {
                m_ptr.m_value = value;
                if (!m_ptr.m_device)
                        *m_ptr.m_pointer = value;
                else
                        CUDA_CHECK_ERROR(cudaMemcpy(m_ptr.m_pointer, &value, sizeof(T),
                                cudaMemcpyHostToDevice), "Failed to copy data to the GPU.");
        }

}; // class Ref

/**
 * \brief A owning or non-owning span around some memory that may be on the host
 * or on the device.
 *
 * \code Span::update()\endcode should be called if any modification is done to
 * the pointed data.
 *
 * \tparam T The type of the span pointer
 */
template<typename T>
class Span {
public:
        using value_type = T;

        Span(uint32_t size, bool device, T *src, bool srcDevice, bool updateOnDestruction = false) :
                m_src(src), m_size(size), m_device(device),
                m_updateOnDestruction(updateOnDestruction) {

                if (device == srcDevice) {
                        m_ptr = src;
                        m_owning = false;
                        return;
                }

                m_owning = true;
                if (!m_device) {
                        m_ptr = new T[m_size];

                        CUDA_CHECK_ERROR(cudaMemcpy(m_ptr, src, m_size * sizeof(T),
                                cudaMemcpyDeviceToHost), "Failed to copy data from the GPU.");

                        return;
                }

                CUDA_CHECK_ERROR(cudaMalloc(&m_ptr, m_size * sizeof(float)),
                        "Failed to allocate memory in the GPU.");
                CUDA_CHECK_ERROR(cudaMemcpy(m_ptr, src, m_size * sizeof(T),
                        cudaMemcpyHostToDevice), "Failed to copy data to the GPU.");
        }

        ~Span()
        {
                if (m_updateOnDestruction)
                        this->update();

                if (m_owning) {
                        if (!m_device)
                                delete[] m_ptr;
                        else
                                CUDA_CHECK_ERROR(cudaFree(m_ptr), "Failed to free GPU memory.");
                }

                m_ptr                 = nullptr;
                m_src                 = nullptr;
                m_owning              = false;
                m_updateOnDestruction = false;
        }

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

        void update()
        {
                if (!m_owning)
                        return;

                if (!m_device)
                        CUDA_CHECK_ERROR(cudaMemcpy(m_src, m_ptr, m_size * sizeof(T),
                                cudaMemcpyHostToDevice), "Failed to copy data to the GPU.");
                else
                        CUDA_CHECK_ERROR(cudaMemcpy(m_src, m_ptr, m_size * sizeof(T),
                                cudaMemcpyDeviceToHost), "Failed to copy data from the GPU.");
        }

private:
        T               *m_src;
        T               *m_ptr;
        uint32_t        m_size;
        bool            m_device;
        bool            m_owning;
        bool            m_updateOnDestruction;

}; // class Span

#pragma once

#include <cuda_runtime.h>

#include <type_traits>
#include <cstddef>

#include <neural-network/utils/Logger.h>
#include <neural-network/CudaBase.h>
#include <neural-network/Base.h>

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
        static_assert(std::is_trivially_copyable_v<T>,
                "Ptr<T> requires T to be trivially copiable. "
                "https://en.cppreference.com/w/cpp/named_req/TriviallyCopyable");

        using value_type = T;
        using difference_type = ptrdiff_t;

        inline Ptr() noexcept : m_pointer(nullptr), m_stream(0), m_device(false) {}

        Ptr(T *ptr, bool device, cudaStream_t stream = 0) noexcept :
                m_pointer(ptr), m_stream(stream), m_device(device) {}

        Ptr(const Ptr &other) noexcept :
                m_pointer(other.m_pointer), m_stream(other.m_stream),
                m_device(other.m_device) {}

        Ptr &operator= (const Ptr &other) noexcept
        {
                if (this == &other)
                        return *this;

                m_pointer = other.m_pointer;
                m_device  = other.m_device;
                m_stream  = other.m_stream;

                return *this;
        }

        Ptr(Ptr &&other) noexcept :
                m_pointer(other.m_pointer), m_stream(other.m_stream),
                m_device(other.m_device) {

                other.m_pointer = nullptr;
        }

        Ptr &operator= (Ptr &&other) noexcept
        {
                if (this == &other)
                        return *this;

                m_pointer = other.m_pointer;
                m_device  = other.m_device;
                m_stream  = other.m_stream;

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

                return { m_pointer, m_device, m_stream };
        }

        [[nodiscard]] constexpr Ref<T> operator[] (uint32_t i)
        {
#ifdef DEBUG_MODE_ENABLED
                if (!m_pointer)
                        throw LOGGER_EX("Cannot index null pointer.");
#endif // DEBUG_MODE_ENABLED

                return { m_pointer + i, m_device, m_stream };
        }

        [[nodiscard]] constexpr Ptr operator+ (uint32_t offset) const noexcept
        {
                return { m_pointer + offset, m_device, m_stream };
        }

        [[nodiscard]] constexpr Ptr operator- (uint32_t offset) const noexcept
        {
                return { m_pointer - offset, m_device, m_stream };
        }

        [[nodiscard]] constexpr difference_type operator- (const Ptr &other) const noexcept
        {
                return m_pointer - other.m_pointer;
        }

        constexpr Ptr &operator+= (uint32_t offset) noexcept
        {
                m_pointer += offset;
                return *this;
        }

        constexpr Ptr &operator-= (uint32_t offset) noexcept
        {
                m_pointer -= offset;
                return *this;
        }

        [[nodiscard]] constexpr Ptr operator++ (int) const noexcept
        {
                return { m_pointer + 1, m_device, m_stream };
        }

        [[nodiscard]] constexpr Ptr operator-- (int) const noexcept
        {
                return { m_pointer - 1, m_device, m_stream };
        }

        constexpr Ptr &operator++ () noexcept
        {
                ++m_pointer;
                return *this;
        }

        constexpr Ptr &operator-- () noexcept
        {
                --m_pointer;
                return *this;
        }

        [[nodiscard]] constexpr T *get() noexcept
        {
                return m_pointer;
        }

        [[nodiscard]] constexpr const T *get() const noexcept
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
        T                   *m_pointer;
        cudaStream_t        m_stream;
        bool                m_device;

}; // class Ptr

/**
 * \brief Reference wrapper around a variable of type \code T\endcode which may
 * be on the host memory or on the device memory.
 *
 * \tparam T The referred type.
 */
template<typename T>
class Ref {
        using HostT = std::conditional_t<sizeof(T) <= 8, T, const T &>;

public:
        using value_type = T;

        Ref(T *pValue, bool device, cudaStream_t stream = 0) :
                m_ptr(pValue, device, stream) {

                if (!pValue)
                        throw LOGGER_EX("Cannot create null reference.");
        }

        Ref(const Ref &other) : m_ptr(other.m_ptr) {}
        Ref &operator= (const Ref &other)
        {
                if (this != std::addressof(other))
                        _set(other);

                return *this;
        }

        Ref(Ref &&other) noexcept : m_ptr(other.m_ptr) {}
        Ref &operator= (Ref &&other) noexcept
        {
                if (this != std::addressof(other))
                        _set(other);

                return *this;
        }

        Ref &operator= (HostT value) requires (!std::is_const_v<T>)
        {
                _set(value);
                return *this;
        }

        [[nodiscard]] constexpr bool operator== (const Ref &other) const noexcept
        {
                return (T)*this == (T)other;
        }

        [[nodiscard]] constexpr bool operator!= (const Ref &other) const noexcept
        {
                return !this->operator==(other);
        }

        [[nodiscard]] constexpr Ptr<T> &operator& () noexcept
        {
                return m_ptr;
        }

        [[nodiscard]] constexpr operator T() const
        {
                return _get();
        }

protected:
        Ptr<T>        m_ptr;

        T _get() const
        {
                if (!m_ptr.m_device)
                        return *m_ptr.m_pointer;

                T value;
                CUDA_CHECK_ERROR(cudaMemcpyAsync(&value, m_ptr.m_pointer, sizeof(T),
                        cudaMemcpyDeviceToHost, m_ptr.m_stream), "Failed to copy data from the GPU.");
                _sync();
                return value;
        }

        void _set(HostT value)
        {
                if (!m_ptr.m_device)
                        *m_ptr.m_pointer = value;
                else
                        CUDA_CHECK_ERROR(cudaMemcpyAsync(m_ptr.m_pointer, &value, sizeof(T),
                                cudaMemcpyHostToDevice, m_ptr.m_stream), "Failed to copy data to the GPU.");
        }

        void _sync() const
        {
                CUDA_CHECK_ERROR(cudaStreamSynchronize(m_ptr.m_stream),
                        "Error synchronizing in Ref<T>.");
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

        Span(
                uint32_t            size,
                bool                device,
                T                   *const src,
                bool                srcDevice,
                bool                updateOnDestruction = false,
                cudaStream_t        stream = 0) :

                m_src(src), m_ptr(nullptr), m_device(device), m_stream(stream),
                m_size(size), m_owning(false), m_updateOnDestruction(updateOnDestruction) {

                if (device == srcDevice) {
                        m_ptr = src;
                        m_owning = false;
                        return;
                }

                m_owning = true;
                if (!m_device) {
                        m_ptr = new T[m_size];

                        CUDA_CHECK_ERROR(cudaMemcpyAsync(m_ptr, src, m_size * sizeof(T),
                                cudaMemcpyDeviceToHost, m_stream), "Failed to copy data from the GPU.");

                        return;
                }

                CUDA_CHECK_ERROR(cudaMallocAsync(&m_ptr, m_size * sizeof(T), m_stream),
                        "Failed to allocate memory in the GPU.");
                CUDA_CHECK_ERROR(cudaMemcpyAsync(m_ptr, src, m_size * sizeof(T),
                        cudaMemcpyHostToDevice, m_stream), "Failed to copy data to the GPU.");
        }

        ~Span()
        {
                if (m_updateOnDestruction)
                        this->update();  // sync here

                _clear();

                m_ptr                 = nullptr;
                m_src                 = nullptr;
                m_owning              = false;
                m_updateOnDestruction = false;
        }

        Span(const Span &other)             = delete;
        Span &operator= (const Span &other) = delete;

        Span(Span &&other) noexcept
        {
                if (this == &other)
                        return;

                m_src                 = other.m_src;
                m_ptr                 = other.m_ptr;
                m_device              = other.m_device;
                m_stream              = other.m_stream;
                m_size                = other.m_size;
                m_owning              = other.m_owning;
                m_updateOnDestruction = other.m_updateOnDestruction;

                other.m_ptr = nullptr;
                other.m_src = nullptr;
        }

        Span &operator= (Span &&other) noexcept
        {
                if (this == &other)
                        return *this;

                if (m_updateOnDestruction)
                        this->update();

                _clear();

                m_src                 = other.m_src;
                m_ptr                 = other.m_ptr;
                m_device              = other.m_device;
                m_stream              = other.m_stream;
                m_size                = other.m_size;
                m_owning              = other.m_owning;
                m_updateOnDestruction = other.m_updateOnDestruction;

                other.m_ptr = nullptr;
                other.m_src = nullptr;

                return *this;
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
                return { m_ptr + i, m_device, m_stream };
        }

        [[nodiscard]] inline Ref<T> begin() const
        {
                return { m_ptr, m_device, m_stream };
        }

        [[nodiscard]] inline Ref<T> end() const
        {
                return { m_ptr + m_size, m_device, m_stream };
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
                        CUDA_CHECK_ERROR(cudaMemcpyAsync(m_src, m_ptr, m_size * sizeof(T),
                                cudaMemcpyHostToDevice, m_stream), "Failed to copy data to the GPU.");
                else
                        CUDA_CHECK_ERROR(cudaMemcpyAsync(m_src, m_ptr, m_size * sizeof(T),
                                cudaMemcpyDeviceToHost, m_stream), "Failed to copy data from the GPU.");

                _sync();
        }

private:
        T                   *m_src;
        T                   *m_ptr;
        bool                m_device;
        cudaStream_t        m_stream;
        uint32_t            m_size;
        bool                m_owning;
        bool                m_updateOnDestruction;

        void _clear()
        {
                if (!m_owning || !m_ptr)
                        return;

                if (!m_device)
                        delete[] m_ptr;
                else
                        CUDA_CHECK_ERROR(cudaFreeAsync(m_ptr, m_stream), "Failed to free GPU memory.");
        }

        void _sync() const
        {
                if (!m_owning)
                        return;

                CUDA_CHECK_ERROR(cudaStreamSynchronize(m_stream), "Error synchronizing in Span<T>.");
        }

}; // class Span

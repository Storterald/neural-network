#pragma once

#include <cuda_runtime.h>

#include <type_traits>
#include <cstddef> // ptrdiff_t

#include <neural-network/utils/logger.h>
#include <neural-network/cuda_base.h>
#include <neural-network/base.h>

namespace nn {

template<typename>
class ptr;

template<typename>
class ref;

template<typename>
class span;

/**
 * \brief Pointer wrapper around a variable of type \code T\endcode which may be
 * on the host memory or on the device memory.
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

        inline ptr() noexcept : m_pointer(nullptr), m_stream(0), m_device(false) {}

        ptr(T *ptr, bool device, cudaStream_t stream = 0) noexcept :
                m_pointer(ptr), m_stream(stream), m_device(device) {}

        ptr(const ptr &other) noexcept :
                m_pointer(other.m_pointer), m_stream(other.m_stream),
                m_device(other.m_device) {}

        ptr &operator= (const ptr &other) noexcept
        {
                if (this == &other)
                        return *this;

                m_pointer = other.m_pointer;
                m_device  = other.m_device;
                m_stream  = other.m_stream;

                return *this;
        }

        ptr(ptr &&other) noexcept :
                m_pointer(other.m_pointer), m_stream(other.m_stream),
                m_device(other.m_device) {

                other.m_pointer = nullptr;
        }

        ptr &operator= (ptr &&other) noexcept
        {
                if (this == &other)
                        return *this;

                m_pointer = other.m_pointer;
                m_device  = other.m_device;
                m_stream  = other.m_stream;

                other.m_pointer = nullptr;

                return *this;
        }

        template<typename other_type>
        [[nodiscard]] constexpr bool operator== (const ptr<other_type> &other) const noexcept
        {
                return m_pointer == other.get() && m_device == other.is_device();
        }

        template<typename other_type>
        [[nodiscard]] constexpr bool operator!= (const ptr<other_type> &other) const noexcept
        {
                return !this->operator==(other);
        }

        /// Warning: Returns true even if location is different
        [[nodiscard]] constexpr bool operator== (const void *other) const noexcept
        {
                return m_pointer == other;
        }

        /// Warning: Returns false even if location is different
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

                return { m_pointer, m_device, m_stream };
        }

        [[nodiscard]] constexpr ref<value_type> operator[] (uint32_t i) const
        {
#ifdef DEBUG_MODE_ENABLED
                if (!m_pointer)
                        throw LOGGER_EX("Cannot index null pointer.");
#endif // DEBUG_MODE_ENABLED

                return { m_pointer + i, m_device, m_stream };
        }

        [[nodiscard]] constexpr ptr operator+ (difference_type offset) const noexcept
        {
                return { m_pointer + offset, m_device, m_stream };
        }

        [[nodiscard]] constexpr ptr operator- (difference_type offset) const noexcept
        {
                return { m_pointer - offset, m_device, m_stream };
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
                return { m_pointer + 1, m_device, m_stream };
        }

        [[nodiscard]] constexpr ptr operator-- (int) noexcept
        {
                --m_pointer;
                return { m_pointer - 1, m_device, m_stream };
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
                return m_device;
        }

        [[nodiscard]] span<value_type> make_span(uint32_t size, bool device, bool update = false)
        {
                return { size, device, m_pointer, m_device, update };
        }

        [[nodiscard]] span<const_value_type> make_span(uint32_t size, bool device, bool update = false) const
        {
                return { size, device, (const_value_type *)m_pointer, m_device, update };
        }

protected:
        value_type          *m_pointer;
        cudaStream_t        m_stream;
        bool                m_device;

}; // class ptr

/**
 * \brief Reference wrapper around a variable of type \code T\endcode which may
 * be on the host memory or on the device memory.
 *
 * \tparam T The referred type.
 */
template<typename T>
class ref {
        using raw_type = std::remove_const_t<T>;
        using arg_type = std::conditional_t<sizeof(raw_type) <= 8, raw_type, const raw_type &>;

public:
        using value_type = T;
        using const_value_type = std::add_const_t<T>;

        ref(T *pValue, bool device, cudaStream_t stream = 0) :
                m_ptr(pValue, device, stream) {

                if (!pValue)
                        throw LOGGER_EX("Cannot create null reference.");
        }

        ref(const ref &other) : m_ptr(other.m_ptr) {}
        ref &operator= (const ref &other)
        {
                if (this != std::addressof(other))
                        _set(other);

                return *this;
        }

        ref(ref &&other) noexcept : m_ptr(other.m_ptr) {}
        ref &operator= (ref &&other) noexcept
        {
                if (this != std::addressof(other))
                        _set(other);

                return *this;
        }

        ref &operator= (arg_type value) requires (!std::is_const_v<T>)
        {
                _set(value);
                return *this;
        }

        [[nodiscard]] inline bool operator== (const ref &other) const
        {
                return _get() == other._get();
        }

        [[nodiscard]] inline bool operator!= (const ref &other) const
        {
                return !this->operator==(other);
        }

        [[nodiscard]] inline ptr<value_type> operator& () const noexcept
        {
                return m_ptr;
        }

        [[nodiscard]] inline operator value_type() const
        {
                return _get();
        }

protected:
        ptr<value_type>        m_ptr;

        value_type _get() const
        {
                if (!m_ptr.m_device)
                        return *m_ptr.m_pointer;

                raw_type value;
                CUDA_CHECK_ERROR(cudaMemcpyAsync(&value, m_ptr.m_pointer, sizeof(T),
                        cudaMemcpyDeviceToHost, m_ptr.m_stream), "Failed to copy data from the GPU.");
                _sync();
                return value;
        }

        void _set(arg_type value)
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
                        "Error synchronizing in ref<T>.");
        }

}; // class ref

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
class span {
        using raw_type = std::remove_const_t<T>;

public:
        using value_type = T;
        using const_value_type = std::add_const_t<T>;

        span(
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
                        m_ptr = new raw_type[m_size]();

                        CUDA_CHECK_ERROR(cudaMemcpyAsync((raw_type *)m_ptr, src, m_size * sizeof(raw_type),
                                cudaMemcpyDeviceToHost, m_stream), "Failed to copy data from the GPU.");

                        return;
                }

                CUDA_CHECK_ERROR(cudaMallocAsync(&m_ptr, m_size * sizeof(raw_type), m_stream),
                        "Failed to allocate memory in the GPU.");
                CUDA_CHECK_ERROR(cudaMemcpyAsync((raw_type *)m_ptr, src, m_size * sizeof(raw_type),
                        cudaMemcpyHostToDevice, m_stream), "Failed to copy data to the GPU.");
        }

        ~span()
        {
                if (m_updateOnDestruction)
                        this->update();  // sync here

                _clear();

                m_ptr                 = nullptr;
                m_src                 = nullptr;
                m_owning              = false;
                m_updateOnDestruction = false;
        }

        span(const span &other)             = delete;
        span &operator= (const span &other) = delete;

        span(span &&other) noexcept
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

        span &operator= (span &&other) noexcept
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
                return { m_ptr + i, m_device, m_stream };
        }

        [[nodiscard]] inline ref<const_value_type> operator[] (uint32_t i) const
        {
                return { m_ptr + i, m_device, m_stream };
        }

        [[nodiscard]] inline ref<value_type> begin()
        {
                return { m_ptr, m_device, m_stream };
        }

        [[nodiscard]] inline ref<const_value_type> begin() const
        {
                return { m_ptr, m_device, m_stream };
        }

        [[nodiscard]] inline ref<value_type> end()
        {
                return { m_ptr + m_size, m_device, m_stream };
        }

        [[nodiscard]] inline ref<const_value_type> end() const
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

        [[nodiscard]] stream stream() const noexcept
        {
                return m_stream;
        }

        void update() requires (!std::is_const_v<value_type>)
        {
                if (!m_owning)
                        return;

                if (!m_device)
                        CUDA_CHECK_ERROR(cudaMemcpyAsync(m_src, m_ptr, m_size * sizeof(value_type),
                                cudaMemcpyHostToDevice, m_stream), "Failed to copy data to the GPU.");
                else
                        CUDA_CHECK_ERROR(cudaMemcpyAsync(m_src, m_ptr, m_size * sizeof(value_type),
                                cudaMemcpyDeviceToHost, m_stream), "Failed to copy data from the GPU.");

                _sync();
        }

        constexpr void update() requires (std::is_const_v<value_type>) {}

private:
        T                   *m_src;
        T                   *m_ptr;
        cudaStream_t        m_stream;
        uint32_t            m_size;
        bool                m_device;
        bool                m_owning;
        bool                m_updateOnDestruction;

        void _clear()
        {
                if (!m_owning || !m_ptr)
                        return;

                if (!m_device)
                        delete[] m_ptr;
                else
                        CUDA_CHECK_ERROR(cudaFreeAsync((raw_type *)m_ptr, m_stream),
                                "Failed to free GPU memory.");
        }

        void _sync() const
        {
                if (!m_owning)
                        return;

                CUDA_CHECK_ERROR(cudaStreamSynchronize(m_stream), "Error synchronizing in span<T>.");
        }

}; // class span

} // namespace nn

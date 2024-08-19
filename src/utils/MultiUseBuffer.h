#pragma once

#include <cstdint>

// Stack allocated continuous buffer
template<uint32_t ...sizes>
struct MultiUseBuffer {
        static_assert(sizeof...(sizes) > 0, "\n\n"
                "The size count must be at least 1.");

private:
        uint32_t offsets[sizeof...(sizes)]{};
        uint8_t data[(sizes + ...)]{};

public:
        static constexpr uint32_t size { (sizes + ...) };

        constexpr MultiUseBuffer()
        {
                for (uint32_t tmp[] { sizes... }, count { 0 }, i { 0 }; i < sizeof...(sizes); count += tmp[i++])
                        offsets[i] = count;
        }

        template<uint32_t i> requires (i < sizeof...(sizes))
        [[nodiscard]] constexpr void *get()
        {
                return data + offsets[i];
        }
};
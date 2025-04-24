#pragma once

#ifndef BUILD_CUDA_SUPPORT

#include "Data.h"

Data::Data(uint32_t size) : m_size(size), m_data(new float[size]()) {}

Data::Data(const Data &other) : m_size(other.m_size), m_data(new float[other.m_size])
{
        // Unified memory is allocated on the CPU by default, so a simple std::memcpy
        // should be faster.
        std::memcpy(m_data, other.m_data, m_size * sizeof(float));
}

Data::Data(Data &&other) noexcept : m_size(other.m_size), m_data(other.m_data)
{
        other.m_size = 0;
        other.m_data = nullptr;
}

Data::~Data()
{
        delete [] m_data;
}

Data &Data::operator= (const Data &other)
{
        if (this == &other)
                return *this;

        // The buffer must be destroyed and remade either if the buffer size is different
        // or the data is in different places.
        if (m_size != other.m_size) {
                // If size doesn't match, overwrite m_size and delete old buffer.
                m_size = other.m_size;
                if (m_data)
                        delete [] m_data;

                m_data = new float [m_size];
        }

        // Once the buffer has the correct size, the data can be copied.
        std::memcpy(m_data, other.m_data, m_size * sizeof(float));

        return *this;
}

Data &Data::operator= (Data &&other) noexcept
{
        if (this == &other)
                return *this;

        delete [] m_data;

        m_size = other.m_size;
        m_data = other.m_data;

        other.m_size = 0;
        other.m_data = nullptr;

        return *this;
}

bool Data::operator== (const Data &other) const
{
        if (m_size != other.m_size)
                return false;

        bool ans = true;
        for (uint32_t i = 0; i < m_size && ans; ++i)
                ans = m_data[i] == other.m_data[i];

        return ans;
}

#endif // BUILD_CUDA_SUPPORT
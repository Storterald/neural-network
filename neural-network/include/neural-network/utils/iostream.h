#pragma once

#include <iostream>

#include <neural-network/types/matrix.h>
#include <neural-network/types/vector.h>

inline std::ostream &operator<< (std::ostream &os, const nn::matrix &mat)
{
        os << '[';
        for (uint32_t i = 0; i < mat.height(); ++i) {
                os << '[';
                for (uint32_t j = 0; j < mat.width() - 1; ++j)
                        os << mat.at(i, j) << ", ";

                os << mat.at(i, mat.width() - 1) << ']';
        }

        os << ']';
        return os;
}

inline std::ostream &operator<< (std::ostream &os, const nn::vector &vec)
{
        os << '[';
        for (uint32_t i = 0; i < vec.size() - 1; ++i)
                os << vec.at(i) << ", ";

        os << vec.at(vec.size() - 1) << ']';
        return os;
}
#pragma once

#include <neural-network/utils/iostream.h>
#include <neural-network/types/matrix.h>
#include <neural-network/types/vector.h>

namespace nn {

template<floating_type _type>
inline void PrintTo(const nn::vector<_type> &vec, std::ostream *os)
{
        *os << vec;
}

template<floating_type _type>
inline void PrintTo(const nn::matrix<_type> &mat, std::ostream *os)
{
        *os << mat;
}

} // namespace nn
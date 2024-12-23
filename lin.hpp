#ifndef LIN_HPP
#define LIN_HPP

#include "cnf.hpp"

namespace mm {
  Matrix add(const Matrix& A, const Matrix& B);
  Matrix sub(const Matrix& A, const Matrix& B);
  Matrix mul(const Matrix& A, const Matrix& B);
  Matrix transpose(const Matrix& A);
} // namespace mm

#endif // LIN_HPP
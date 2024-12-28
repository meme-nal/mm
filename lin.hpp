#ifndef LIN_HPP
#define LIN_HPP

#include "cnf.hpp"

namespace mm {
  /*******************
  // Matrix basic operations
  *******************/
  Matrix add(const Matrix& A, const Matrix& B);
  Matrix sub(const Matrix& A, const Matrix& B);
  Matrix mul(const Matrix& A, const Matrix& B);
  Matrix transpose(const Matrix& A);

  /*******************
  // Tensor basic operations
  *******************/
  Matrix add(const Matrix& A, const Matrix& B);
  Matrix sub(const Matrix& A, const Matrix& B);

  /*******************
  // Convolution operations, F - filter
  *******************/
  //Matrix Conv1d(const Matrix& X, const Matrix& F); 
  //Matrix Conv2d(const Matrix& X, const Matrix& F);
  //Tensor Conv3d(const Tensor& X, const Tensor& F);

  /*******************
  // print usitls
  *******************/
  std::ostream& operator << (std::ostream& out, const Matrix& m);
  //std::ostream& operator << (std::ostream& out, const Tensor& t);

} // namespace mm

#endif // LIN_HPP
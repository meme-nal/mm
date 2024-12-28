#include "lin.hpp"

namespace mm {

  Matrix add(const Matrix& A, const Matrix& B) {
    // check if matrices are the same size
    if (A.size() != B.size() && A[0].size() != B[0].size()) {
      throw std::runtime_error("Matrices are not the same size");
    }

    Matrix res(A.size(), std::vector<float>(A[0].size()));

    for (size_t i {0}; i < res.size(); ++i) {
      for (size_t j {0}; j < res[0].size(); ++j) {
        res[i][j] = A[i][j] + B[i][j];
      }
    }

    return res;
  }


  Matrix sub(const Matrix& A, const Matrix& B) {
    // check if matrices are the same size
    if (A.size() != B.size() && A[0].size() != B[0].size()) {
      throw std::runtime_error("Matrices are not the same size");
    }

    Matrix res(A.size(), std::vector<float>(A[0].size()));

    for (size_t i {0}; i < res.size(); ++i) {
      for (size_t j {0}; j < res[0].size(); ++j) {
        res[i][j] = A[i][j] - B[i][j];
      }
    }

    return res;
  }


  Matrix transpose(const Matrix& A) {
    size_t newRows {A[0].size()};
    size_t newCols {A.size()};

    Matrix res(newRows, std::vector<float>(newCols));

    for (size_t i {0}; i < newRows; ++i) {
      for (size_t j {0}; j < newCols; ++j) {
        res[i][j] = A[j][i];
      }
    }

    return res;
  }


  Matrix mul(const Matrix& A, const Matrix& B) {
    // check if matrices are proper size
    if (A[0].size() != B.size()) {
      throw std::runtime_error("Cannot multiply matrices");
    }

    Matrix res(A.size(), std::vector<float>(B[0].size()));

    size_t inner_shape {B.size()};

    for (size_t i {0}; i < res.size(); ++i) {
      for (size_t j {0}; j < res[0].size(); ++j) {
        float sum {0.f};
        for (size_t k {0}; k < inner_shape; ++k) {
          sum += A[i][k] * B[k][j];
        }
        res[i][j] = sum;
      }
    }

    return res;
  }

  Tensor add(const Tensor& A, const Tensor& B) {
    // check if tensors are the same size
    if (    A.size() != B.size() &&
         A[0].size() != B[0].size() &&
      A[0][0].size() != B[0][0].size()) {
      throw std::runtime_error("Tensors are not the same size");
    }

    Tensor res(A.size(), Matrix(A[0].size(), std::vector<float>(A[0][0].size())));

    for (size_t i {0}; i < res.size(); ++i) {
      for (size_t j {0}; j < res[0].size(); ++j) {
        for (size_t k {0}; k < res[0][0].size(); ++k) {
          res[i][j][k] = A[i][j][k] + B[i][j][k];
        }
      }
    }

    return res;
  }


  Tensor sub(const Tensor& A, const Tensor& B) {
    // check if tensors are the same size
    if (    A.size() != B.size() &&
         A[0].size() != B[0].size() &&
      A[0][0].size() != B[0][0].size()) {
      throw std::runtime_error("Tensors are not the same size");
    }

    Tensor res(A.size(), Matrix(A[0].size(), std::vector<float>(A[0][0].size())));

    for (size_t i {0}; i < res.size(); ++i) {
      for (size_t j {0}; j < res[0].size(); ++j) {
        for (size_t k {0}; k < res[0][0].size(); ++k) {
          res[i][j][k] = A[i][j][k] - B[i][j][k];
        }
      }
    }

    return res;
  }


  std::ostream& operator << (std::ostream& out, const Matrix& m) {
    out << "[ ";
    for (size_t i {0}; i < m.size(); ++i) {
      out << "[ ";
      for (size_t j {0}; j < m[0].size(); ++j) {
        out << m[i][j] << ' ';
      }
      out << ']';
    }
    out << " ]";
    return out;
  }

} // namespace mm
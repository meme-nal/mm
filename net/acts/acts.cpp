#include "acts.hpp"

namespace mm {
  namespace net {
    namespace acts {

      Vector ReLU(Vector Z) {
        Vector a(Z.size(), 0.f);
        for (size_t i {0}; i < Z.size(); ++i) {
          a[i] = (Z[i] < 0.f ? 0.f : Z[i]);
        }
        return a;
      }


      Matrix ReLU(Matrix Z) {
        Matrix a(Z.size(), Vector(Z[0].size(), 0.f));
        for (size_t i {0}; i < a.size(); ++i) {
          for (size_t j {0}; j < a[0].size(); ++j) {
            a[i][j] = (Z[i][j] < 0.f ? 0.f : Z[i][j]);
          }
        }
        return a;
      }


      Tensor ReLU(Tensor Z) {
        Tensor a(Z.size(), Matrix(Z[0].size(), Vector(Z[0][0].size(), 0.f)));
        for (size_t i {0}; i < a.size(); ++i) {
          for (size_t j {0}; j < a[0].size(); ++j) {
            for (size_t k {0}; k < a[0][0].size(); ++k) {
              a[i][j][k] = (Z[i][j][k] < 0.f ? 0.f : Z[i][j][k]);
            }
          }
        }
        return a;
      }


      Vector Sigmoid(Vector Z) {
        Vector a(Z.size(), 0.f);
        for (size_t i {0}; i < Z.size(); ++i) {
          a[i] = 1.f / (1.f + std::exp(-Z[i]));
        }
        return a;
      }


      Matrix Sigmoid(Matrix Z) {
        Matrix a(Z.size(), Vector(Z[0].size(), 0.f));
        for (size_t i {0}; i < a.size(); ++i) {
          for (size_t j {0}; j < a[0].size(); ++j) {
            a[i][j] = 1.f / (1.f + std::exp(-Z[i][j]));
          }
        }
        return a;
      }


      Tensor Sigmoid(Tensor Z) {
        Tensor a(Z.size(), Matrix(Z[0].size(), Vector(Z[0][0].size(), 0.f)));
        for (size_t i {0}; i < a.size(); ++i) {
          for (size_t j {0}; j < a[0].size(); ++j) {
            for (size_t k {0}; k < a[0][0].size(); ++k) {
              a[i][j][k] = 1.f / (1.f + std::exp(-Z[i][j][k]));
            }
          }
        }
        return a;
      }


      Vector LeakyReLU(Vector Z, const float negative_slope) {
        Vector a(Z.size(), 0.f);
        for (size_t i {0}; i < Z.size(); ++i) {
          a[i] = (Z[i] < 0.f ? negative_slope * Z[i] : Z[i]);
        }
        return a;
      }

      Matrix LeakyReLU(Matrix Z, const float negative_slope) {
        Matrix a(Z.size(), Vector(Z[0].size(), 0.f));
        for (size_t i {0}; i < a.size(); ++i) {
          for (size_t j {0}; j < a[0].size(); ++j) {
            a[i][j] = (Z[i][j] < 0.f ? negative_slope * Z[i][j] : Z[i][j]);
          }
        }
        return a;
      }

      Tensor LeakyReLU(Tensor Z, const float negative_slope) {
        Tensor a(Z.size(), Matrix(Z[0].size(), Vector(Z[0][0].size(), 0.f)));
        for (size_t i {0}; i < a.size(); ++i) {
          for (size_t j {0}; j < a[0].size(); ++j) {
            for (size_t k {0}; k < a[0][0].size(); ++k) {
              a[i][j][k] = (Z[i][j][k] < 0.f ? negative_slope * Z[i][j][k] : Z[i][j][k]);
            }
          }
        }
        return a;
      }


      Vector Linear(Vector Z) {
        Vector a(Z.size(), 0.f);
        for (size_t i {0}; i < Z.size(); ++i) {
          a[i] = Z[i];
        }
        return a;
      }

      Matrix Linear(Matrix Z) {
        Matrix a(Z.size(), Vector(Z[0].size(), 0.f));
        for (size_t i {0}; i < a.size(); ++i) {
          for (size_t j {0}; j < a[0].size(); ++j) {
            a[i][j] = Z[i][j];
          }
        }
        return a;
      }

      Tensor Linear(Tensor Z) {
        Tensor a(Z.size(), Matrix(Z[0].size(), Vector(Z[0][0].size(), 0.f)));
        for (size_t i {0}; i < a.size(); ++i) {
          for (size_t j {0}; j < a[0].size(); ++j) {
            for (size_t k {0}; k < a[0][0].size(); ++k) {
              a[i][j][k] = Z[i][j][k];
            }
          }
        }
        return a;
      }


      Vector ELU(Vector Z, const float alpha) {
        Vector a(Z.size(), 0.f);
        for (size_t i {0}; i < Z.size(); ++i) {
          a[i] = (Z[i] < 0.f ? alpha * (std::exp(Z[i]) - 1) : Z[i]);
        }
        return a;
      }

      Matrix ELU(Matrix Z, const float alpha) {
        Matrix a(Z.size(), Vector(Z[0].size(), 0.f));
        for (size_t i {0}; i < a.size(); ++i) {
          for (size_t j {0}; j < a[0].size(); ++j) {
            a[i][j] = (Z[i][j] < 0.f ? alpha * (std::exp(Z[i][j]) - 1) : Z[i][j]);
          }
        }
        return a;
      }

      Tensor ELU(Tensor Z, const float alpha) {
        Tensor a(Z.size(), Matrix(Z[0].size(), Vector(Z[0][0].size(), 0.f)));
        for (size_t i {0}; i < a.size(); ++i) {
          for (size_t j {0}; j < a[0].size(); ++j) {
            for (size_t k {0}; k < a[0][0].size(); ++k) {
              a[i][j][k] = (Z[i][j][k] < 0.f ? alpha * (std::exp(Z[i][j][k]) - 1) : Z[i][j][k]);
            }
          }
        }
        return a;
      }


      Vector Tanh(Vector Z) {
        Vector a(Z.size(), 0.f);
        for (size_t i {0}; i < Z.size(); ++i) {
          a[i] = std::tanh(Z[i]);
        }
        return a;
      }

      Matrix Tanh(Matrix Z) {
        Matrix a(Z.size(), Vector(Z[0].size(), 0.f));
        for (size_t i {0}; i < a.size(); ++i) {
          for (size_t j {0}; j < a[0].size(); ++j) {
            a[i][j] = std::tanh(Z[i][j]);
          }
        }
        return a;
      }

      Tensor Tanh(Tensor Z) {
        Tensor a(Z.size(), Matrix(Z[0].size(), Vector(Z[0][0].size(), 0.f)));
        for (size_t i {0}; i < a.size(); ++i) {
          for (size_t j {0}; j < a[0].size(); ++j) {
            for (size_t k {0}; k < a[0][0].size(); ++k) {
              a[i][j][k] = std::tanh(Z[i][j][k]);
            }
          }
        }
        return a;
      }


      Vector Swish(Vector Z, const float beta) {
        Vector a(Z.size(), 0.f);
        for (size_t i {0}; i < Z.size(); ++i) {
          a[i] = Z[i] / (1.f + std::exp(-beta*Z[i]));
        }
        return a;
      }

      Matrix Swish(Matrix Z, const float beta) {
        Matrix a(Z.size(), Vector(Z[0].size(), 0.f));
        for (size_t i {0}; i < a.size(); ++i) {
          for (size_t j {0}; j < a[0].size(); ++j) {
            a[i][j] = Z[i][j] / (1.f + std::exp(-beta*Z[i][j]));
          }
        }
        return a;
      }

      Tensor Swish(Tensor Z, const float beta) {
        Tensor a(Z.size(), Matrix(Z[0].size(), Vector(Z[0][0].size(), 0.f)));
        for (size_t i {0}; i < a.size(); ++i) {
          for (size_t j {0}; j < a[0].size(); ++j) {
            for (size_t k {0}; k < a[0][0].size(); ++k) {
              a[i][j][k] = Z[i][j][k] / (1.f + std::exp(-beta*Z[i][j][k]));
            }
          }
        }
        return a;
      }
    } // namespace acts
  } // namespace net
} // namespace mm
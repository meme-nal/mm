#include "loss.hpp"

namespace mm {
  namespace net {
    namespace loss {
      Scalar MAE(Vector pr, Vector gt) {
        Scalar loss {0.f};
        for (size_t i {0}; i < pr.size(); ++i) {
          loss += std::abs(pr[i] - gt[i]);
        }
        return loss / pr.size();
      }

      Scalar MAE(Matrix pr, Matrix gt) {
        Scalar total_loss {0.f};
        for (size_t i {0}; i < pr.size(); ++i) {
          Scalar loss {0.f};
          for (size_t j {0}; j < pr[0].size(); ++j) {
            loss += std::abs(pr[i][j] - gt[i][j]);
          }
          total_loss += loss / pr[0].size();
        }
        return total_loss / pr.size();
      }

      Scalar MSE(Vector pr, Vector gt) {
        Scalar loss {0.f};
        for (size_t i {0}; i < pr.size(); ++i) {
          loss += (pr[i] - gt[i]) * (pr[i] - gt[i]);
        }
        return loss / pr.size();
      }

      Scalar MSE(Matrix pr, Matrix gt) {
        Scalar total_loss {0.f};
        for (size_t i {0}; i < pr.size(); ++i) {
          Scalar loss {0.f};
          for (size_t j {0}; j < pr[0].size(); ++j) {
            loss += (pr[i][j] - gt[i][j]) * (pr[i][j] - gt[i][j]);
          }
          total_loss += loss / pr[0].size();
        }
        return total_loss / pr.size();
      }

    } // namespace loss
  } // namespace net
} // namespace mm
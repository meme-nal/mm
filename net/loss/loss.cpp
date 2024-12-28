#include "loss.hpp"

namespace mm {
  namespace net {
    namespace loss {

      float MAE(const Matrix& pr, const Matrix& gt) {
        float total_loss {0.f};
        for (size_t i {0}; i < pr.size(); ++i) {
          float loss {0.f};
          for (size_t j {0}; j < pr[0].size(); ++j) {
            loss += std::abs(pr[i][j] - gt[i][j]);
          }
          total_loss += loss / pr[0].size();
        }
        return total_loss / pr.size();
      }


      float MAE(const Tensor& pr, const Tensor& gt) {
        float total_loss {0.f};
        for (size_t i {0}; i < pr.size(); ++i) {
          total_loss += MAE(pr[i], gt[i]);
        }
        return total_loss / pr.size();
      }


      float MSE(const Matrix& pr, const Matrix& gt) {
        float total_loss {0.f};
        for (size_t i {0}; i < pr.size(); ++i) {
          float loss {0.f};
          for (size_t j {0}; j < pr[0].size(); ++j) {
            loss += (pr[i][j] - gt[i][j]) * (pr[i][j] - gt[i][j]);
          }
          total_loss += loss / pr[0].size();
        }
        return total_loss / pr.size();
      }


      float MSE(const Tensor& pr, const Tensor& gt) {
        float total_loss {0.f};
        for (size_t i {0}; i < pr.size(); ++i) {
          total_loss += MSE(pr[i], gt[i]);
        }
        return total_loss / pr.size();
      }

    } // namespace loss
  } // namespace net
} // namespace mm
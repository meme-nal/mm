#ifndef LOSS_HPP
#define LOSS_HPP

#include "../../cnf.hpp"

namespace mm {
  namespace net {
    namespace loss {
      /*******************
      // MAE function
      *******************/
      float MAE(const Matrix& pr, const Matrix& gt);
      float MAE(const Tensor& pr, const Tensor& gt);

      /*******************
      // MSE function
      *******************/
      float MSE(const Matrix& pr, const Matrix& gt);
      float MSE(const Tensor& pr, const Tensor& gt);

    } // namespace loss
  } // namespace net
} // namespace mm

#endif // LOSS_HPP
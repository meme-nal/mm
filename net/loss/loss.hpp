#ifndef LOSS_HPP
#define LOSS_HPP

#include "../../cnf.hpp"

namespace mm {
  namespace net {
    namespace loss {
      /*******************
      // MAE function
      *******************/
      Scalar MAE(Vector pr, Vector gt);
      Scalar MAE(Matrix pr, Matrix gt);

      /*******************
      // MSE function
      *******************/
      Scalar MSE(Vector pr, Vector gt);
      Scalar MSE(Matrix pr, Matrix gt);

    } // namespace loss
  } // namespace net
} // namespace mm

#endif // LOSS_HPP
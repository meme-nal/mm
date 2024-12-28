#ifndef ACTS_HPP
#define ACTS_HPP

#include "../../cnf.hpp"


namespace mm {
  namespace net {
    namespace acts {
      /*******************
      // ReLU function
      *******************/
      Matrix ReLU(Matrix);
      Tensor ReLU(Tensor);

      /*******************
      // Sigmoid function
      *******************/
      Matrix Sigmoid(Matrix);
      Tensor Sigmoid(Tensor);

      /*******************
      // LeakyReLU function
      *******************/
      Matrix LeakyReLU(Matrix, const float negative_slope=1e-2f);
      Tensor LeakyReLU(Tensor, const float negative_slope=1e-2f);

      /*******************
      // Linear function
      *******************/
      Matrix Linear(Matrix);
      Tensor Linear(Tensor);

      /*******************
      // ELU function
      *******************/
      Matrix ELU(Matrix, const float alpha=1.f);
      Tensor ELU(Tensor, const float alpha=1.f);

      /*******************
      // Tanh function
      *******************/
      Matrix Tanh(Matrix);
      Tensor Tanh(Tensor);

      /*******************
      // Swish function
      *******************/
      Matrix Swish(Matrix, const float beta=1.f);
      Tensor Swish(Tensor, const float beta=1.f);


    } // namespace acts
  } // namespace net
} // namespace mm

#endif // ACTS_HPP
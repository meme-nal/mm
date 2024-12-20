#ifndef ACTS_HPP
#define ACTS_HPP

#include "../../cnf.hpp"


namespace mm {
  namespace net {
    namespace acts {
      /*******************
      // ReLU function
      *******************/
      Vector ReLU(Vector);
      Matrix ReLU(Matrix);
      Tensor ReLU(Tensor);

      /*******************
      // Sigmoid function
      *******************/
      Vector Sigmoid(Vector);
      Matrix Sigmoid(Matrix);
      Tensor Sigmoid(Tensor);

      /*******************
      // LeakyReLU function
      *******************/
      Vector LeakyReLU(Vector, const float negative_slope=1e-2f);
      Matrix LeakyReLU(Matrix, const float negative_slope=1e-2f);
      Tensor LeakyReLU(Tensor, const float negative_slope=1e-2f);

      /*******************
      // Linear function
      *******************/
      Vector Linear(Vector);
      Matrix Linear(Matrix);
      Tensor Linear(Tensor);

      /*******************
      // ELU function
      *******************/
      Vector ELU(Vector, const float alpha=1.f);
      Matrix ELU(Matrix, const float alpha=1.f);
      Tensor ELU(Tensor, const float alpha=1.f);

      /*******************
      // Tanh function
      *******************/
      Vector Tanh(Vector);
      Matrix Tanh(Matrix);
      Tensor Tanh(Tensor);

      /*******************
      // Swish function
      *******************/
      Vector Swish(Vector, const float beta=1.f);
      Matrix Swish(Matrix, const float beta=1.f);
      Tensor Swish(Tensor, const float beta=1.f);


    } // namespace acts
  } // namespace net
} // namespace mm

#endif // ACTS_HPP
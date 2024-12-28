#include "dense.hpp"
#include "../acts/acts.hpp"
#include "../../lin.hpp"

namespace mm {
  namespace net {
    Dense::Dense(const json& layer_node) {
      _act = layer_node["act"].get<std::string>();
      _winit = layer_node["winit"].get<std::string>();

      _W = Matrix(layer_node["W_shape"][0].get<size_t>(), std::vector<float>(layer_node["W_shape"][1].get<size_t>()));
      _B = Matrix(layer_node["B_shape"][0].get<size_t>(), std::vector<float>(layer_node["B_shape"][1].get<size_t>()));

      winit(_winit);
    }

    void Dense::winit(const std::string type) {
      if (type == "one") {
        for (size_t i {0}; i < _W.size(); ++i) {
          for (size_t j {0}; j < _W[0].size(); ++j) {
            _W[i][j] = 1.f;
          }
        }
        for (size_t i {0}; i < _B.size(); ++i) {
          for (size_t j {0}; j < _B[0].size(); ++j) {
            _B[i][j] = 1.f;
          }
        }
      } else if (type == "zero") {
        return; // weights are equal zero by default
      } else {
        throw std::runtime_error("Incorrect type of weights initialization");
      }
    }

    Matrix Dense::forward(Matrix X) {
      Matrix Z = add(mul(transpose(this->_W), X), this->_B);
      this->_Z = Z;

      Matrix a;
      if (this->_act == "ReLU") {
        a = acts::ReLU(Z);
      } else if (this->_act == "Linear") {
        a = acts::Linear(Z);
      } else {
        throw std::runtime_error("Incorrect type of activation function");
      }

      this->_a = a;

      return a; 
    }

    void Dense::backward() {
      std::cout << "BACKPROPAGATION\n";
    }
    
  } // namespace net
} // namespace mm
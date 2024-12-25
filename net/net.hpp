#ifndef NET_HPP
#define NET_HPP

#include "acts/acts.hpp"
#include "loss/loss.hpp"

#include "layers/layer.hpp"
#include "layers/dense.hpp"

namespace mm {
  namespace net {
    class net {
    private:
      std::vector<ILayer*> _layers;
      std::string _loss;
      Scalar _lr;
      size_t _miniBatchSize;
      std::string _optimizer;

    public:
      net(const json& net_cnf);
      ~net() {
        for (const auto* layer : this->_layers) {
          delete layer;
        }
      }

    public:
      Matrix forward(Matrix);
      void backward();
    };

    ILayer* makeLayerFactory(const json& arch_node, const std::string& layer_name);
  } // net
} // mm

#endif // NET_HPP
#include "net.hpp"

namespace mm {
  namespace net {
    std::shared_ptr<ILayer> makeLayerFactory(const json& arch_node, const std::string& layer_name) {
      const std::string type = arch_node[layer_name]["type"].get<std::string>();
      if (type == "Dense") {
        return std::make_shared<Dense>(arch_node[layer_name]);
      } else {
        throw std::runtime_error("Incorrect type of a layer");
      }
    }

    net::net(const json& net_cnf) {
      const size_t layersNum {net_cnf["net"]["arch"]["to_use"].size()};

      _lr = net_cnf["net"]["optim"]["lr"].get<float>();
      _loss = net_cnf["net"]["optim"]["loss"].get<std::string>();
      _optimizer = net_cnf["net"]["optim"]["optimizer"].get<std::string>();
      _miniBatchSize = net_cnf["net"]["optim"]["miniBatchSize"].get<size_t>();

      const auto& layers = net_cnf["net"]["arch"]["to_use"];

      for (size_t i {0}; i < layersNum; ++i) {
        _layers.emplace_back(makeLayerFactory(net_cnf["net"]["arch"], layers[i]));
      }
    }

    Matrix net::forward(Matrix X) {
      for (const auto& layer : this->_layers) {
        X = layer->forward(X);
      }

      return X;
    }

    void net::backward() {
      for (const auto& layer : this->_layers) {
        layer->backward();
      }
    }
    
  } // namespace net
} // namespace mm
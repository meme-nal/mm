#ifndef DENSE_HPP
#define DENSE_HPP

#include "layer.hpp"

namespace mm {
  namespace net {
    class Dense : public ILayer {
    private:
      Matrix _W;
      Matrix _B;
      Matrix _Z;
      Matrix _a;
      std::string _act;
      std::string _winit;

    public:
      Dense(const json& layer_node);
    
    private:
      void winit(const std::string type);

    public:
      Matrix forward(Matrix) override;
      void backward() override;
    };

  } // namespace net
} // namespace mm

#endif // DENSE_HPP
#ifndef LAYER_HPP
#define LAYER_HPP

#include "../../cnf.hpp"

namespace mm {
  namespace net {
    class ILayer {
    public:
      ILayer() = default;
      virtual ~ILayer() {};

    public:
      virtual Matrix forward(Matrix) = 0;
      virtual void backward() = 0; 
    };
    
  } // namespace net
} // namespace mm

#endif // LAYER_HPP
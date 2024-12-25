#ifndef CNF_HPP
#define CNF_HPP

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <iostream>
#include <vector>
#include <cmath>

namespace mm {
  using Scalar = float;
  using Vector = std::vector<float>;
  using Matrix = std::vector<std::vector<float>>;
  using Tensor = std::vector<std::vector<std::vector<float>>>;

  enum LAYER_TYPE {
    DENSE
  };

  namespace net {
    namespace acts {}
    namespace loss {}
    namespace optim {}
    namespace winit {}
  } // namespace net
} // namespace net

#endif // CNF_HPP
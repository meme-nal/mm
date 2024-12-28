#ifndef CNF_HPP
#define CNF_HPP

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

namespace mm {
  using Matrix = std::vector<std::vector<float>>;
  using Tensor = std::vector<Matrix>;

  namespace net {
    namespace acts {}
    namespace loss {}
    namespace optim {}
    namespace winit {}
  } // namespace net
} // namespace net

#endif // CNF_HPP
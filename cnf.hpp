#ifndef CNF_HPP
#define CNF_HPP

#include <iostream>
#include <vector>
#include <cmath>

namespace mm {
  using Vector = std::vector<float>;
  using Matrix = std::vector<std::vector<float>>;
  using Tensor = std::vector<std::vector<std::vector<float>>>;

  namespace net {
    namespace acts {}
    namespace loss {}
    namespace optim {}
    namespace winit {}
  } // namespace net
} // namespace net

#endif // CNF_HPP
#include "acts.hpp"

namespace mm {
  namespace net {
    namespace acts {

      Vector ReLU(Vector Z) {
        Vector a(Z.size(), 0.f);
        for (size_t i {0}; i < Z.size(); ++i) {
          a[i] = (Z[i] < 0.f ? 0.f : Z[i]);
        }
        return a;
      }

    } // namespace acts
  } // namespace net
} // namespace mm
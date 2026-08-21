#include "driver_version.h"

#include <cstdlib>
#include <iostream>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

} // namespace

int main() {
  const int newer_routes[] = {13040, 14000};
  require(lupine_highest_common_driver_version(13030, newer_routes, 2) == 13030,
          "client API level did not cap negotiated driver version");

  const int mixed_routes[] = {13030, 12090, 13010};
  require(lupine_highest_common_driver_version(13030, mixed_routes, 3) == 12090,
          "negotiation did not select the highest common route version");

  require(lupine_highest_common_driver_version(13030, nullptr, 0) == 0,
          "route-less negotiation unexpectedly succeeded");

  const int invalid_route[] = {13030, 0};
  require(lupine_highest_common_driver_version(13030, invalid_route, 2) == 0,
          "invalid route version unexpectedly negotiated");
  return 0;
}

#include "mlx_tagged_simulator.h"
#include <iomanip>
#include <iostream>
int main() {
  using namespace mlx::tagged;
  const unsigned values[] = {0,0x8000,0x3c00,0xbc00,0x7c00,0xfc00,0x7e01,0xfe01};
  for (unsigned a : values) for (unsigned b : values) {
    std::array<Vector,3> args{}; args[0].fill(a); args[1].fill(b); args[2].fill(0);
    std::cout << std::hex << a << ' ' << b;
    for (unsigned op : {3,9,6,5,4,2}) std::cout << ' ' << arithmetic(op,args,4)[0];
    std::cout << '\n';
  }
}

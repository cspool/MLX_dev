#include "Vmlx_tagged_fu.h"
#include "mlx_tagged_simulator.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#ifndef MLX_FU_LANES
#define MLX_FU_LANES 32
#endif
double sc_time_stamp() { return 0; }
using mlx::tagged::Vector;
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
template <class Port> Vector read_vec(const Port &port) {
  Vector value{};
  for (unsigned lane=0; lane<MLX_FU_LANES; ++lane) {
    if constexpr (std::is_integral_v<Port>) value[lane]=port>>(lane*16);
    else value[lane]=port[lane/2]>>((lane%2)*16);
  }
  return value;
}
template <class Port> void write_vec(Port &port, const Vector &value) {
  if constexpr (std::is_integral_v<Port>) {
    port=0;
    for (unsigned lane=0; lane<MLX_FU_LANES; ++lane) port|=uint64_t(value[lane])<<(lane*16);
  } else for (unsigned lane=0; lane<MLX_FU_LANES; lane+=2)
    port[lane/2]=uint32_t(value[lane]) | uint32_t(value[lane+1])<<16;
}
struct Bench {
  Vmlx_tagged_fu dut;
  uint64_t cycle=0, requests=0, responses=0, held=0, last_issue=0;
  unsigned ii;
  explicit Bench(unsigned interval): ii(interval) {
    dut.clk=0; dut.rst_n=0; dut.request_valid_i=0; dut.response_ready_i=0;
    dut.operation_i=3; dut.request_tag_i=0; dut.compute_ii_i=ii;
    write_vec(dut.operand_a_i,Vector{}); write_vec(dut.operand_b_i,Vector{}); write_vec(dut.operand_c_i,Vector{});
    edge(); dut.rst_n=1; dut.eval(); cycle=0;
  }
  void edge() { dut.clk=0; dut.eval(); dut.clk=1; dut.eval(); dut.clk=0; dut.eval(); ++cycle; }
  void transact(unsigned op, const std::array<Vector,3> &operands, unsigned delay=0, bool invalid=false) {
    const uint64_t tag=0xfedcba9800000000ULL | requests;
    dut.operation_i=op; dut.request_tag_i=tag; dut.request_valid_i=1; dut.response_ready_i=0;
    write_vec(dut.operand_a_i,operands[0]); write_vec(dut.operand_b_i,operands[1]); write_vec(dut.operand_c_i,operands[2]);
    dut.eval();
    while (!dut.request_ready_o) { edge(); require(cycle<1000000, "FU request watchdog"); }
    if (requests) require(cycle-last_issue>=ii, "FU violated initiation interval");
    last_issue=cycle;
    const unsigned latency=op==2?4:op==3||op==9?3:op==5?8:op==6?12:1;
    const auto due=cycle+latency;
    const auto expected=invalid?Vector{}:mlx::tagged::arithmetic(op,operands,MLX_FU_LANES);
    ++requests; edge();
    dut.request_valid_i=0;
    // Poison the live buses: results/ownership must have been captured at issue.
    Vector poison{}; poison.fill(0x7eaa);
    write_vec(dut.operand_a_i,poison); write_vec(dut.operand_b_i,poison); write_vec(dut.operand_c_i,poison);
    dut.request_tag_i=~tag; dut.operation_i=15;
    while (true) {
      dut.response_ready_i=cycle>=due+delay; dut.eval();
      require(bool(dut.response_valid_o)==(cycle>=due), "FU response latency differs from native contract");
      require(dut.busy_o && !dut.request_ready_o, "FU accepted work before draining the in-flight operation");
      if (dut.response_valid_o) {
        require(dut.response_tag_o==tag, "FU response tag changed");
        require(bool(dut.response_error_o)==invalid, "FU opcode error mismatch");
        if (!invalid) require(read_vec(dut.response_data_o)==expected, "FU vector data mismatch");
        if (!dut.response_ready_i) ++held;
      }
      const bool done=dut.response_valid_o && dut.response_ready_i;
      edge();
      if (done) { ++responses; break; }
      require(cycle<1000000, "FU response watchdog");
    }
    require(!dut.busy_o && !dut.response_valid_o, "completed FU retained stale valid state");
  }
};
int main(int argc, char **argv) {
  try {
    require(argc==2, "usage: fu-test output.json");
    uint64_t state=0x93db8fa10e7364c5ULL, checked=0, held=0;
    auto random=[&]() { state^=state<<13; state^=state>>7; state^=state<<17; return uint16_t(state); };
    for (unsigned ii:{1,17}) {
      Bench bench(ii);
      for (unsigned n=0;n<256;++n) {
        std::array<Vector,3> args{};
        for (auto &v:args) for (unsigned lane=0;lane<MLX_FU_LANES;++lane) v[lane]=random();
        for (unsigned op:{2,3,4,5,6,7,9}) bench.transact(op,args,n%5);
      }
      for (unsigned op:{0,1,8,10,15}) bench.transact(op,{},3,true);
      checked+=bench.responses; held+=bench.held;
      require(bench.requests==bench.responses, "FU transaction conservation failed");
      bench.dut.operation_i=6; bench.dut.request_valid_i=1; bench.dut.response_ready_i=0;
      bench.dut.eval(); while (!bench.dut.request_ready_o) bench.edge();
      bench.edge(); bench.dut.request_valid_i=0; bench.edge();
      bench.dut.rst_n=0; bench.edge(); bench.dut.rst_n=1; bench.dut.eval();
      require(!bench.dut.busy_o && !bench.dut.response_valid_o && bench.dut.request_ready_o,
              "reset did not discard pending FU ownership");
      bench.requests=bench.responses=0;
      bench.transact(3,{}); // No stale data/tag after reset.
    }
    Json::Value report(Json::objectValue);
    report["classification"]="functional_vector_fu_rtl_not_full_M4";
    report["lanes"]=MLX_FU_LANES; report["transactions"]=Json::UInt64(checked);
    report["held_response_cycles"]=Json::UInt64(held); report["mismatches"]=0;
    report["initiation_intervals"].append(1); report["initiation_intervals"].append(17);
    std::ofstream out(argv[1]); out<<report<<'\n'; require(bool(out), "cannot write FU report");
    std::cout<<"MLX_TAGGED_FU_PASS lanes="<<MLX_FU_LANES<<" transactions="<<checked<<'\n';
  } catch (const std::exception &error) {
    std::cerr<<"MLX_TAGGED_FU_FAIL: "<<error.what()<<'\n'; return 1;
  }
}

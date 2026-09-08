#include "Vmlx_tagged_fp16_lane.h"
#include "mlx_tagged_simulator.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>

double sc_time_stamp() { return 0; }
int main(int argc, char **argv) {
  try {
    if (argc != 3) throw std::invalid_argument("usage: fp16-test output.json random-triples");
    Vmlx_tagged_fp16_lane dut;
    std::array<mlx::tagged::Vector,3> operands{};
    std::map<unsigned,uint64_t> counts;
    Json::Value failures(Json::arrayValue);
    uint64_t mismatches = 0;
    auto check = [&](unsigned op, uint16_t a, uint16_t b=0, uint16_t c=0) {
      dut.op_i=op; dut.a_i=a; dut.b_i=b; dut.c_i=c; dut.eval();
      operands[0][0]=a; operands[1][0]=b; operands[2][0]=c;
      const auto expected=mlx::tagged::arithmetic(op,operands,4)[0];
      ++counts[op];
      if (dut.illegal_o || dut.result_o!=expected) {
        ++mismatches;
        if (failures.size()<20) {
          Json::Value f(Json::objectValue);
          f["op"]=op; f["a"]=a; f["b"]=b; f["c"]=c; f["expected"]=expected; f["actual"]=dut.result_o;
          failures.append(f);
          std::cerr<<"FP16_MISMATCH op="<<op<<std::hex<<" a="<<a<<" b="<<b<<" c="<<c
                   <<" expected="<<expected<<" actual="<<dut.result_o<<std::dec<<'\n';
        }
      }
    };
    // Unary EXP is checked for every input encoding, including special values.
    for (unsigned a=0;a<65536;++a) check(5,a);
    std::cout<<"EXP_SWEEP_COMPLETE mismatches="<<mismatches<<std::endl;
    const unsigned anchors[] = {0,0x8000,1,0x8001,0x3ff,0x400,0x3555,0x3bff,0x3c00,0x3c01,
        0xbc00,0x7bff,0xfbff,0x7c00,0xfc00,0x7e01,0xfe01};
    for (unsigned a=0;a<65536;++a) for (unsigned b:anchors) {
      for (unsigned op:{3,4,6,9}) check(op,a,b);
      check(2,a,b,uint16_t(a)^0x8000);
    }
    std::cout<<"ANCHOR_SWEEP_COMPLETE mismatches="<<mismatches<<std::endl;
    uint64_t state=0x2d358dccaa6c78a5ULL;
    auto random = [&]() { state^=state<<13; state^=state>>7; state^=state<<17; return uint16_t(state); };
    const unsigned trials=std::stoul(argv[2]);
    for (unsigned n=0;n<trials;++n) {
      const uint16_t a=random(), b=random(), c=random();
      for (unsigned op:{2,3,4,6,9}) check(op,a,b,c);
    }
    check(2,0x3c01,0x3c01,0xbc02); // Must not contract into fused FMA.
    check(4,0x8000,0); // Equal-zero MAX chooses the left bit pattern.
    for (unsigned op:{0,1,8,10,11,12,13,14,15}) {
      dut.op_i=op; dut.eval();
      if (!dut.illegal_o) throw std::runtime_error("invalid FU opcode accepted");
    }
    Json::Value report(Json::objectValue);
    report["classification"]="scalar_fp16_functional_rtl_not_full_M4";
    report["exp_input_encodings"]=65536; report["anchor_count"]=unsigned(sizeof(anchors)/sizeof(*anchors));
    report["random_triples"]=trials; report["random_seed"]="2d358dccaa6c78a5";
    for (const auto &[op,count]:counts) report["checked"][mlx::tagged::opcode_name(op)]=Json::UInt64(count);
    report["mismatches"]=Json::UInt64(mismatches); report["examples"]=failures;
    std::ofstream out(argv[1]); out<<report<<'\n';
    if (!out) throw std::runtime_error("cannot save FP16 report");
    if (mismatches) return 1;
    std::cout<<"MLX_TAGGED_FP16_PASS\n";
  } catch (const std::exception &error) {
    std::cerr<<"MLX_TAGGED_FP16_FAIL: "<<error.what()<<'\n'; return 2;
  }
}

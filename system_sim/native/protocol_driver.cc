#include "device.h"
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#ifdef MLX_DPI_DRIVER
#include "wire_device.h"
double mlx_wire_time=0;
double sc_time_stamp() { return mlx_wire_time; }
#endif

using mlx::system::Inputs;
class Bus {
public:
#ifdef MLX_DPI_DRIVER
  WireDevice device;
#else
  mlx::system::Device device;
#endif
  std::map<uint64_t,uint64_t> memory;
  uint64_t cycles = 0;
  unsigned delay, period;
  bool stall_responses = false;
  std::optional<std::pair<unsigned,uint64_t>> response;
  Bus(unsigned latency, unsigned ready_period) : delay(latency), period(ready_period) {}
  bool step(Inputs command = {}) {
    if (cycles > 1000000) throw std::runtime_error("protocol watchdog");
    command.response_ready = !stall_responses;
    command.memory_ready = cycles % period == 0;
    if (pending && pending->due == cycles) {
      command.memory_response_valid = true; command.memory_response_data = pending->data; pending.reset();
    }
    const auto out = device.eval(command);
    if (out.response_valid && command.response_ready) response = {out.response_rd, out.response_data};
    if (out.memory_valid && command.memory_ready) {
      if (pending) throw std::runtime_error("multiple outstanding requests");
      if (out.memory_size != 3 || out.memory_mask != 255 || out.memory_tag != 0)
        throw std::runtime_error("incorrect memory transaction");
      if (out.memory_write) memory[out.memory_address] = out.memory_data;
      pending = Pending{cycles + delay, out.memory_write ? 0 : memory[out.memory_address]};
    }
    device.tick(command); ++cycles;
    return command.command_valid && out.command_ready;
  }
  uint64_t command(uint8_t funct, uint64_t first=0, uint64_t second=0, bool xd=false) {
    Inputs cmd;
    cmd.command_valid=true; cmd.funct=funct; cmd.rs1=first; cmd.rs2=second; cmd.command_xd=xd; cmd.rd=7;
    response.reset();
    while (!step(cmd)) {}
    if (xd) {
      while (!response) step();
      if (response->first != 7) throw std::runtime_error("incorrect response rd");
      return response->second;
    }
    return 0;
  }
private:
  struct Pending { uint64_t due, data; };
  std::optional<Pending> pending;
};

int main(int argc, char **argv) {
  try {
    if (argc < 4) throw std::invalid_argument("usage: mlx-device-test image.hex program.json result.json [delay] [period] [first-half]");
    const unsigned delay=argc>4?std::stoul(argv[4]):3, period=argc>5?std::stoul(argv[5]):1;
    if (!delay || !period) throw std::invalid_argument("positive timing required");
    std::ifstream source(argv[2]); Json::Value raw; source >> raw;
    const auto program=mlx::tagged::Program::parse(raw);
    Bus bus(delay,period);
    constexpr uint64_t input=0x80001000, output=0x80010000;
    unsigned vectors=0;
    for (unsigned a=0;a<program.hardware.spm_vectors;++a) if (program.input_valid[a]) {
      for (unsigned beat=0;beat<program.hardware.lanes/4;++beat) {
        uint64_t value=0;
        for (unsigned lane=0;lane<4;++lane) value |= uint64_t(program.inputs[a][beat*4+lane]) << (lane*16);
        bus.memory[input+vectors*program.hardware.lanes*2+beat*8]=value;
      }
      ++vectors;
    }
    if (argc>6) bus.memory[input]=(bus.memory[input]&~uint64_t(65535))|std::stoul(argv[6]);
    if (bus.command(3,14,0,true)!=mlx::system::ABI_MAGIC) throw std::runtime_error("ABI mismatch");
    bus.stall_responses=true;
    Inputs status;
    status.command_valid=true; status.command_xd=true; status.funct=3; status.rs1=14; status.rd=9;
    if (!bus.step(status)) throw std::runtime_error("status not accepted");
    for (unsigned n=0;n<5;++n) {
      const auto out=bus.device.eval(Inputs{});
      if (!out.response_valid || out.response_rd!=9 || out.response_data!=mlx::system::ABI_MAGIC)
        throw std::runtime_error("response changed under backpressure");
      bus.step();
    }
    bus.stall_responses=false; bus.step();
    const auto config_start=bus.cycles;
    std::ifstream image(argv[1]);
    if (!image) throw std::runtime_error("cannot open image");
    uint64_t address,word;
    while (image>>std::hex>>address>>word) bus.command(0,word,address);
    if (!image.eof()) throw std::runtime_error("invalid image record");
    const auto config_cycles=bus.cycles-config_start, launch_start=bus.cycles;
    bus.command(1,input,output);
    const auto wait=bus.command(2,0,0,true);
    const auto launch_cycles=bus.cycles-launch_start;
#ifdef MLX_DPI_DRIVER
    Json::Value result(Json::objectValue);
    result["error"]=unsigned(bus.command(3,15,0,true));
    result["system_cycles"]=Json::UInt64(bus.command(3,1,0,true));
#else
    auto result=bus.device.report();
#endif
    result["host_kind"]="bus_protocol_driver_not_riscv";
    result["host_config_cycles"]=Json::UInt64(config_cycles);
    result["host_launch_wait_cycles"]=Json::UInt64(launch_cycles);
    result["wait_status"]=Json::UInt64(wait);
    std::set<unsigned> slots(program.outputs.begin(),program.outputs.end());
    unsigned index=0;
    for (auto a:slots) {
      auto &vector=result["host_outputs"][std::to_string(a)]; vector=Json::Value(Json::arrayValue);
      for (unsigned beat=0;beat<program.hardware.lanes/4;++beat) {
        const auto value=bus.memory[output+index*program.hardware.lanes*2+beat*8];
        for (unsigned lane=0;lane<4;++lane) vector.append(unsigned((value>>(lane*16))&65535));
      }
      ++index;
    }
    for (unsigned s=0;s<=16;++s) result["queried_status"].append(Json::UInt64(bus.command(3,s,0,true)));
    std::ofstream destination(argv[3]); destination<<result<<'\n';
    if (!destination) throw std::runtime_error("cannot write result");
    std::cout<<"MLX_DEVICE_TEST_COMPLETE error="<<result["error"].asUInt()
             <<" system="<<result["system_cycles"].asUInt64()<<'\n';
  } catch (const std::exception &error) {
    std::cerr<<"MLX_DEVICE_TEST_ERROR: "<<error.what()<<'\n'; return 2;
  }
}

#include "device.h"
#include <svdpi.h>

#include <cstdlib>
#include <fstream>
#include <iostream>

#ifndef MLX_NATIVE_BUILD_ID
#define MLX_NATIVE_BUILD_ID "standalone-unbound"
#endif

extern "C" void *mlx_system_create(unsigned int address_bits) {
  try {
    mlx::tagged::Timing timing;
    const char *trace=std::getenv("MLX_NATIVE_TRACE");
    timing.trace=trace && std::string(trace)=="1";
    const char *serial=std::getenv("MLX_NATIVE_SERIAL");
    timing.overlap=!(serial && std::string(serial)=="1");
    return new mlx::system::Device(address_bits,timing);
  } catch (const std::exception &error) {
    std::cerr<<"MLX_NATIVE_CREATE_ERROR: "<<error.what()<<'\n'; return nullptr;
  }
}
extern "C" void mlx_system_reset(void *model) {
  if (model) static_cast<mlx::system::Device *>(model)->reset();
}
extern "C" void mlx_system_destroy(void *model) {
  if (!model) return;
  auto *device=static_cast<mlx::system::Device *>(model);
  auto report=device->report();
  report["build_identity"]=MLX_NATIVE_BUILD_ID;
  const char *path=std::getenv("MLX_NATIVE_REPORT");
  if (path) {
    std::ofstream file(path); file<<report<<'\n';
    if (!file) std::cerr<<"MLX_NATIVE_REPORT_ERROR: "<<path<<'\n';
  }
  std::cout<<"MLX_NATIVE_DEVICE_FINAL system="<<report["system_cycles"].asUInt64()
           <<" dma="<<report["dma_cycles"].asUInt64()<<" kernel="<<report["kernel_cycles"].asUInt64()
           <<" error="<<report["error"].asUInt()<<'\n';
  delete device;
}
extern "C" void mlx_system_tick(void *model, svBit valid, unsigned int funct, svBit xd,
    unsigned int rd, unsigned long long rs1, unsigned long long rs2, unsigned int privilege,
    svBit response_ready, svBit memory_ready, svBit memory_response_valid,
    unsigned int memory_response_tag, unsigned long long memory_response_data) {
  if (!model) return;
  mlx::system::Inputs in;
  in.command_valid=valid; in.funct=funct; in.command_xd=xd; in.rd=rd;
  in.rs1=rs1; in.rs2=rs2; in.privilege=privilege; in.response_ready=response_ready;
  in.memory_ready=memory_ready; in.memory_response_valid=memory_response_valid;
  in.memory_response_tag=memory_response_tag; in.memory_response_data=memory_response_data;
  static_cast<mlx::system::Device *>(model)->tick(in);
}
extern "C" void mlx_system_eval(void *model, unsigned long long /*epoch*/, svBit reset,
    unsigned int funct, svBit response_ready, svBit *command_ready, svBit *response_valid,
    unsigned int *rd, unsigned long long *data, svBit *busy, svBit *memory_valid,
    svBit *memory_write, unsigned long long *address, unsigned long long *write_data,
    unsigned int *tag, unsigned int *size, unsigned int *mask, unsigned int *privilege) {
  mlx::system::Outputs out;
  if (model && !reset) {
    mlx::system::Inputs in; in.funct=funct; in.response_ready=response_ready;
    out=static_cast<mlx::system::Device *>(model)->eval(in);
  }
  *command_ready=out.command_ready; *response_valid=out.response_valid; *rd=out.response_rd;
  *data=out.response_data; *busy=out.busy; *memory_valid=out.memory_valid;
  *memory_write=out.memory_write; *address=out.memory_address; *write_data=out.memory_data;
  *tag=out.memory_tag; *size=out.memory_size; *mask=out.memory_mask; *privilege=out.privilege;
}

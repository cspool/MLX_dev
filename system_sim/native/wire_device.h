#pragma once
#include "device.h"
#include "VMLXNativeRoCCBlackBox.h"
extern double mlx_wire_time;

// Signal-level driver of the DPI bridge, separate from the direct C++ protocol test.
class WireDevice {
public:
  WireDevice() {
    model.reset=1;
    for (unsigned i=0;i<3;++i) { model.clock=0; ++mlx_wire_time; model.eval(); model.clock=1; ++mlx_wire_time; model.eval(); }
    model.reset=0;
  }
  ~WireDevice() { model.final(); }
  mlx::system::Outputs eval(const mlx::system::Inputs &in) {
    set_inputs(in); if (model.clock) ++mlx_wire_time; model.clock=0; model.eval();
    mlx::system::Outputs out;
    out.command_ready=model.cmd_ready; out.response_valid=model.resp_valid;
    out.response_rd=model.resp_rd; out.response_data=model.resp_data; out.busy=model.busy;
    out.memory_valid=model.mem_req_valid; out.memory_write=model.mem_req_cmd==1;
    out.memory_address=model.mem_req_addr; out.memory_data=model.mem_req_data;
    out.memory_tag=model.mem_req_tag; out.memory_size=model.mem_req_size;
    out.memory_mask=model.mem_req_mask; out.privilege=model.mem_req_dprv;
    return out;
  }
  void tick(const mlx::system::Inputs &in) { set_inputs(in); if (!model.clock) ++mlx_wire_time; model.clock=1; model.eval(); }
private:
  VMLXNativeRoCCBlackBox model;
  void set_inputs(const mlx::system::Inputs &in) {
    model.cmd_valid=in.command_valid; model.cmd_funct=in.funct; model.cmd_xd=in.command_xd;
    model.cmd_rd=in.rd; model.cmd_rs1=in.rs1; model.cmd_rs2=in.rs2; model.cmd_dprv=in.privilege;
    model.resp_ready=in.response_ready; model.mem_req_ready=in.memory_ready;
    model.mem_resp_valid=in.memory_response_valid; model.mem_resp_tag=in.memory_response_tag;
    model.mem_resp_data=in.memory_response_data;
  }
};

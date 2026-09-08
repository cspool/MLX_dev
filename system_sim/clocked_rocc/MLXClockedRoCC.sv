`timescale 1ns/1ps
// Simulation-only DPI signal bridge. There is no PE/FU/scheduler RTL here.
module MLXClockedRoCCBlackBox #(
    parameter BACKEND=3, XLEN=64, ADDR_BITS=40, TAG_BITS=6, CMD_BITS=5,
    parameter SIZE_BITS=3, DATA_BITS=64, DATA_BYTES=8
) (
    input wire clock, reset,
    output reg cmd_ready,
    input wire cmd_valid,
    input wire [6:0] cmd_funct,
    input wire cmd_xd,
    input wire [4:0] cmd_rd,
    input wire [XLEN-1:0] cmd_rs1, cmd_rs2,
    input wire [1:0] cmd_dprv,
    input wire resp_ready,
    output reg resp_valid,
    output reg [4:0] resp_rd,
    output reg [XLEN-1:0] resp_data,
    input wire mem_req_ready,
    output reg mem_req_valid,
    output reg [ADDR_BITS-1:0] mem_req_addr,
    output reg [TAG_BITS-1:0] mem_req_tag,
    output reg [CMD_BITS-1:0] mem_req_cmd,
    output reg [SIZE_BITS-1:0] mem_req_size,
    output reg mem_req_signed, mem_req_phys,
    output reg [1:0] mem_req_dprv,
    output reg [DATA_BITS-1:0] mem_req_data,
    output reg [DATA_BYTES-1:0] mem_req_mask,
    input wire mem_resp_valid,
    input wire [TAG_BITS-1:0] mem_resp_tag,
    input wire [DATA_BITS-1:0] mem_resp_data,
    output reg busy
);
  import "DPI-C" function chandle mlx_clocked_rocc_create(input int unsigned address_bits, input int unsigned tag_bits);
  import "DPI-C" function void mlx_clocked_rocc_reset(input chandle model);
  import "DPI-C" function void mlx_clocked_rocc_destroy(input chandle model);
  import "DPI-C" function void mlx_clocked_rocc_tick(
      input chandle model, input bit command_valid, input int unsigned funct,
      input bit xd, input int unsigned rd, input longint unsigned rs1,
      input longint unsigned rs2, input int unsigned privilege, input bit response_ready,
      input bit memory_ready, input bit memory_response_valid,
      input int unsigned memory_response_tag, input longint unsigned memory_response_data);
  import "DPI-C" function void mlx_clocked_rocc_eval(
      input chandle model, input longint unsigned epoch, input bit reset_active,
      input int unsigned funct, input bit response_ready,
      output bit command_ready, output bit response_valid, output int unsigned rd,
      output longint unsigned data, output bit device_busy,
      output bit memory_valid, output bit memory_write, output longint unsigned address,
      output longint unsigned write_data, output int unsigned tag,
      output int unsigned size, output int unsigned mask, output int unsigned privilege);

  chandle model;
  longint unsigned epoch=0;
  bit cr, rv, db, mv, mw;
  int unsigned rd_full, tag_full, size_full, mask_full, privilege_full;
  longint unsigned data_full, address_full, write_full;
  initial begin
    if (BACKEND != 3 || XLEN != 64 || DATA_BITS != 64 || DATA_BYTES != 8)
      $fatal(1,"unsupported native model interface");
    model = mlx_clocked_rocc_create(ADDR_BITS,TAG_BITS);
    if (model == null) $fatal(1,"native model construction failed");
  end
  always @(posedge clock) begin
    if (reset) mlx_clocked_rocc_reset(model);
    else mlx_clocked_rocc_tick(model,cmd_valid,32'(cmd_funct),cmd_xd,32'(cmd_rd),cmd_rs1,cmd_rs2,
        32'(cmd_dprv),resp_ready,mem_req_ready,mem_resp_valid,32'(mem_resp_tag),mem_resp_data);
    epoch <= epoch + 1;
  end
  always @* begin
    // epoch makes C++ state changes visible to the simulator's combinational scheduler.
    mlx_clocked_rocc_eval(model,epoch,reset,32'(cmd_funct),resp_ready,cr,rv,rd_full,data_full,db,
                    mv,mw,address_full,write_full,tag_full,size_full,mask_full,privilege_full);
    cmd_ready=cr; resp_valid=rv; resp_rd=rd_full[4:0]; resp_data=data_full; busy=db;
    mem_req_valid=mv; mem_req_addr=address_full[ADDR_BITS-1:0]; mem_req_tag=TAG_BITS'(tag_full);
    mem_req_cmd=CMD_BITS'(mw ? 1 : 0); mem_req_size=SIZE_BITS'(size_full);
    mem_req_signed=0; mem_req_phys=0; mem_req_dprv=privilege_full[1:0];
    mem_req_data=write_full; mem_req_mask=DATA_BYTES'(mask_full);
  end
  final mlx_clocked_rocc_destroy(model);
endmodule


`timescale 1ns/1ps
// One bounded compute operation in flight. Arithmetic is functional RTL;
// this conservative implementation captures a combinational result at issue
// and holds it for the declared service latency. It is not a pipelined-Fmax claim.
module mlx_tagged_fu #(
    parameter integer LANES = 32,
    parameter integer VECTOR_BITS = LANES * 16,
    parameter integer TAG_BITS = 64
) (
    input wire clk, rst_n,
    input wire request_valid_i,
    output wire request_ready_o,
    input wire [3:0] operation_i,
    input wire [VECTOR_BITS-1:0] operand_a_i, operand_b_i, operand_c_i,
    input wire [TAG_BITS-1:0] request_tag_i,
    input wire [31:0] compute_ii_i,
    output wire response_valid_o,
    input wire response_ready_i,
    output reg [VECTOR_BITS-1:0] response_data_o,
    output reg [TAG_BITS-1:0] response_tag_o,
    output reg response_error_o,
    output wire busy_o
);
  reg busy_q;
  reg [3:0] remaining_q;
  reg [31:0] cooldown_q;
  wire [VECTOR_BITS-1:0] computed;
  wire [LANES-1:0] illegal;
  genvar lane;
  initial begin
    if (LANES < 4 || LANES > 32 || (LANES & (LANES-1)) != 0 || VECTOR_BITS != LANES * 16)
      $fatal(1, "unsupported tagged FU geometry");
  end
  generate
    for (lane=0; lane<LANES; lane=lane+1) begin : lanes
      localparam integer PARTNER = lane ^ 1;
      wire [15:0] a = operation_i == 7 ? operand_a_i[PARTNER*16 +: 16] : operand_a_i[lane*16 +: 16];
      mlx_tagged_fp16_lane #(.TRANSCENDENTAL(lane < LANES/4 ? 1 : 0)) arithmetic (
          .op_i(operation_i), .a_i(a), .b_i(operand_b_i[lane*16 +: 16]), .c_i(operand_c_i[lane*16 +: 16]),
          .result_o(computed[lane*16 +: 16]), .illegal_o(illegal[lane]));
    end
  endgenerate
  function automatic [3:0] latency;
    input [3:0] op;
    begin
      case (op)
        2: latency=4;
        3, 9: latency=3;
        5: latency=8;
        6: latency=12;
        default: latency=1;
      endcase
    end
  endfunction
  assign busy_o = busy_q;
  assign request_ready_o = rst_n && !busy_q && cooldown_q <= 1;
  assign response_valid_o = rst_n && busy_q && remaining_q <= 1;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      busy_q<=0; remaining_q<=0; cooldown_q<=0;
      response_data_o<=0; response_tag_o<=0; response_error_o<=0;
    end else begin
      if (cooldown_q != 0) cooldown_q<=cooldown_q-32'd1;
      if (busy_q) begin
        if (remaining_q > 1) remaining_q<=remaining_q-4'd1;
        else if (response_ready_i) busy_q<=0;
      end
      if (request_valid_i && request_ready_o) begin
        busy_q<=1; remaining_q<=latency(operation_i); cooldown_q<=compute_ii_i;
        response_data_o<=computed; response_tag_o<=request_tag_i;
        response_error_o<=|illegal || compute_ii_i == 0;
      end
    end
  end
endmodule

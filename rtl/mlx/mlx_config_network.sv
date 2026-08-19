`timescale 1ns/1ps

module mlx_config_network #(
    parameter WORD_BITS = 64,
    parameter INST_DEPTH = 32,
    parameter ADDR_BITS = 5
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 cfg_valid_i,
    input  wire [ADDR_BITS-1:0] cfg_addr_i,
    input  wire [WORD_BITS-1:0] cfg_word_i,
    output wire                 cfg_ready_o,
    input  wire [ADDR_BITS-1:0] fetch_addr_i,
    output wire [WORD_BITS-1:0] fetch_word_o,
    output reg                  configured_o
);
  reg [WORD_BITS-1:0] instruction_mem [0:INST_DEPTH-1];
  assign cfg_ready_o = 1'b1;
  assign fetch_word_o = instruction_mem[fetch_addr_i];

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      configured_o <= 1'b0;
    end else if (cfg_valid_i) begin
      instruction_mem[cfg_addr_i] <= cfg_word_i;
      configured_o <= 1'b1;
    end
  end
endmodule

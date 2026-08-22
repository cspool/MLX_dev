`timescale 1ns/1ps

(* blackbox *)
module mlx_register_file (
    input wire clk,
    input wire rst_n,
    input wire [3:0] read_addr_a_i,
    input wire [3:0] read_addr_b_i,
    input wire [3:0] read_addr_c_i,
    output wire [511:0] read_data_a_o,
    output wire [511:0] read_data_b_o,
    output wire [511:0] read_data_c_o,
    input wire write_enable_i,
    input wire [3:0] write_addr_i,
    input wire [511:0] write_data_i
);
endmodule

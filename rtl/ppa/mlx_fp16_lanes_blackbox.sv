`timescale 1ns/1ps

(* blackbox *)
module mlx_fp16_alu_lane (
    input wire [3:0] op_i,
    input wire [15:0] a_i,
    input wire [15:0] b_i,
    input wire [15:0] c_i,
    output wire [15:0] result_o
);
endmodule

(* blackbox *)
module mlx_fp16_reduced_lane (
    input wire [3:0] op_i,
    input wire [15:0] a_i,
    input wire [15:0] b_i,
    input wire [15:0] c_i,
    output wire [15:0] result_o
);
endmodule

`timescale 1ns/1ps
module mlx_tagged_fp16_lane #(
    parameter integer TRANSCENDENTAL = 1
) (
    input wire [3:0] op_i,
    input wire [15:0] a_i, b_i, c_i,
    output reg [15:0] result_o,
    output wire illegal_o
);
  `include "mlx_fp16_functions.svh"
  wire [15:0] multiplied = fp_mul(a_i, b_i);
  wire [15:0] added = fp_add(op_i == 2 ? multiplied : a_i, op_i == 2 ? c_i : b_i);
  wire [15:0] exponential, divided;
  generate
    if (TRANSCENDENTAL != 0) begin : nonlinear
      mlx_tagged_exp exponential_unit(.enable_i(op_i == 5), .a_i(a_i), .result_o(exponential));
      assign divided = fp_div(a_i, b_i);
    end else begin : passthrough
      assign exponential = a_i;
      assign divided = a_i;
    end
  endgenerate
  assign illegal_o = !(op_i == 2 || op_i == 3 || op_i == 4 || op_i == 5 || op_i == 6 || op_i == 7 || op_i == 9);
  always @* begin
    case (op_i)
      2, 3: result_o = added;
      4: result_o = fp_max(a_i, b_i);
      5: result_o = exponential;
      6: result_o = divided;
      7: result_o = a_i; // Vector wrapper supplies lane-xor-1 as a_i for shuffle.
      9: result_o = multiplied;
      default: result_o = 0;
    endcase
  end
endmodule

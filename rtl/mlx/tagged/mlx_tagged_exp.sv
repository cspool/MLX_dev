`timescale 1ns/1ps
// exp(x) = 2^n * 2^(j/64) * exp(r), 0 <= r < ln(2)/64.
// Q40 range reduction, degree-4 Taylor polynomial, then binary32->binary16
// rounding to match the registered native expf-based contract. No output LUT.
module mlx_tagged_exp(input wire enable_i, input wire [15:0] a_i, output reg [15:0] result_o);
  `include "mlx_fp16_functions.svh"
  `include "mlx_exp2_fraction.svh"
  function automatic [47:0] qmul;
    input [47:0] a, b;
    reg [95:0] product;
    reg [55:0] rounded;
    begin
      product = a * b; rounded = product[95:40];
      if (product[39] && (product[38:0] != 0 || rounded[0])) rounded = rounded + 56'd1;
      qmul = rounded[47:0];
    end
  endfunction
  reg [47:0] magnitude, r, polynomial, scaled;
  reg signed [47:0] x, y;
  // Q40 reduction deliberately drops 40 fractional bits and the redundant
  // high sign-extension bits; the finite input path is bounded to (-32, 16).
  /* verilator lint_off UNUSED */
  reg signed [95:0] product;
  /* verilator lint_on UNUSED */
  reg [5:0] table_index;
  reg [63:0] float_significand;
  integer exponent, high, shift;
  always @* begin
    result_o = 0; magnitude = 0; r = 0; polynomial = 0; scaled = 0;
    x = 0; y = 0; product = 0; table_index = 0; float_significand = 0;
    exponent = 0; high = 0; shift = 0;
    if (enable_i) begin
      if (fp_nan(a_i[14:0])) result_o = 16'h7e00;
      else if (!a_i[15] && a_i[14:0] >= 15'h4c00) result_o = 16'h7c00; // x >= 16
      else if (a_i[15] && a_i[14:0] >= 15'h5000) result_o = 0; // x <= -32
      else begin
        if (a_i[14:10] == 0) magnitude = {38'd0, a_i[9:0]} << 16;
        else magnitude = {37'd0, 1'b1, a_i[9:0]} << ({27'd0, a_i[14:10]} + 15);
        x = a_i[15] ? -$signed(magnitude) : $signed(magnitude);
        product = x * 48'sh0171547652b8; // RNE(log2(e) * 2^40)
        y = product[87:40];
        exponent = {{24{y[47]}}, y[47:40]};
        table_index = y[39:34];
        r = qmul({14'd0, y[33:0]}, 48'h00b17217f7d2);
        polynomial = qmul(48'h000aaaaaaaab, r) + 48'h002aaaaaaaab;
        polynomial = qmul(polynomial, r) + 48'h008000000000;
        polynomial = qmul(polynomial, r) + 48'h010000000000;
        polynomial = qmul(polynomial, r) + 48'h010000000000;
        scaled = qmul(exp2_fraction(table_index), polynomial);
        high = leading_bit({16'd0, scaled});
        shift = high - 23;
        float_significand = round_shift({16'd0, scaled}, shift);
        result_o = fp_pack(1'b0, float_significand, exponent - 40 + shift);
      end
    end
  end
endmodule

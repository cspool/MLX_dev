`timescale 1ns/1ps

module mlx_fp16_alu_lane (
    input  wire [3:0]  op_i,
    input  wire [15:0] a_i,
    input  wire [15:0] b_i,
    input  wire [15:0] c_i,
    output reg  [15:0] result_o
);
  localparam OP_ADD = 4'd3;
  localparam OP_FMA = 4'd2;
  localparam OP_MAX = 4'd4;
  localparam OP_EXP = 4'd5;
  localparam OP_DIV = 4'd6;
  localparam OP_MUL = 4'd9;

  function [15:0] fp16_mul;
    input [15:0] lhs;
    input [15:0] rhs;
    reg sign;
    reg [10:0] lhs_mant;
    reg [10:0] rhs_mant;
    reg [21:0] product;
    reg [9:0] fraction;
    integer exponent;
    begin
      if ((lhs[14:0] == 15'd0) || (rhs[14:0] == 15'd0)) begin
        fp16_mul = 16'd0;
      end else if ((lhs[14:10] == 5'h1f) || (rhs[14:10] == 5'h1f)) begin
        fp16_mul = {lhs[15] ^ rhs[15], 5'h1f, 10'd0};
      end else begin
        sign = lhs[15] ^ rhs[15];
        lhs_mant = {1'b1, lhs[9:0]};
        rhs_mant = {1'b1, rhs[9:0]};
        product = lhs_mant * rhs_mant;
        exponent = {27'd0, lhs[14:10]} + {27'd0, rhs[14:10]} - 15;
        if (product[21]) begin
          exponent = exponent + 1;
          fraction = product[20:11];
        end else begin
          fraction = product[19:10];
        end
        if (exponent <= 0)
          fp16_mul = 16'd0;
        else if (exponent >= 31)
          fp16_mul = {sign, 5'h1f, 10'd0};
        else
          fp16_mul = {sign, exponent[4:0], fraction};
      end
    end
  endfunction

  function [15:0] fp16_add;
    input [15:0] lhs;
    input [15:0] rhs;
    reg [15:0] large_value;
    reg [15:0] small_value;
    reg [14:0] large_mant;
    reg [14:0] small_mant;
    reg [14:0] magnitude;
    reg sign;
    integer exponent;
    integer shift;
    integer index;
    begin
      if (lhs[14:0] == 15'd0) begin
        fp16_add = rhs;
      end else if (rhs[14:0] == 15'd0) begin
        fp16_add = lhs;
      end else if ((lhs[14:10] == 5'h1f) || (rhs[14:10] == 5'h1f)) begin
        fp16_add = 16'h7c00;
      end else begin
        if ({lhs[14:0]} >= {rhs[14:0]}) begin
          large_value = lhs;
          small_value = rhs;
        end else begin
          large_value = rhs;
          small_value = lhs;
        end
        exponent = {27'd0, large_value[14:10]};
        shift = {27'd0, large_value[14:10]} - {27'd0, small_value[14:10]};
        large_mant = {1'b0, 1'b1, large_value[9:0], 3'b000};
        small_mant = {1'b0, 1'b1, small_value[9:0], 3'b000};
        if (shift > 13)
          small_mant = 15'd0;
        else
          small_mant = small_mant >> shift;
        sign = large_value[15];
        if (large_value[15] == small_value[15])
          magnitude = large_mant + small_mant;
        else
          magnitude = large_mant - small_mant;
        if (magnitude[14]) begin
          magnitude = magnitude >> 1;
          exponent = exponent + 1;
        end else begin
          for (index = 0; index < 11; index = index + 1) begin
            if ((magnitude[13] == 1'b0) && (magnitude != 0)) begin
              magnitude = magnitude << 1;
              exponent = exponent - 1;
            end
          end
        end
        if ((magnitude == 0) || (exponent <= 0))
          fp16_add = 16'd0;
        else if (exponent >= 31)
          fp16_add = {sign, 5'h1f, 10'd0};
        else
          fp16_add = {sign, exponent[4:0], magnitude[12:3]};
      end
    end
  endfunction

  function [15:0] fp16_max;
    input [15:0] lhs;
    input [15:0] rhs;
    begin
      if (lhs[15] != rhs[15])
        fp16_max = lhs[15] ? rhs : lhs;
      else if (!lhs[15])
        fp16_max = (lhs[14:0] >= rhs[14:0]) ? lhs : rhs;
      else
        fp16_max = (lhs[14:0] <= rhs[14:0]) ? lhs : rhs;
    end
  endfunction

  function [15:0] fp16_div;
    input [15:0] lhs;
    input [15:0] rhs;
    reg sign;
    reg [10:0] lhs_mant;
    reg [10:0] rhs_mant;
    reg [21:0] quotient;
    integer exponent;
    begin
      if ((lhs[14:0] == 0) || (rhs[14:0] == 0)) begin
        fp16_div = (rhs[14:0] == 0) ? 16'h7c00 : 16'd0;
      end else begin
        sign = lhs[15] ^ rhs[15];
        lhs_mant = {1'b1, lhs[9:0]};
        rhs_mant = {1'b1, rhs[9:0]};
        quotient = ({11'd0, lhs_mant} << 10) / {11'd0, rhs_mant};
        exponent = {27'd0, lhs[14:10]} - {27'd0, rhs[14:10]} + 15;
        if (quotient < 1024) begin
          quotient = quotient << 1;
          exponent = exponent - 1;
        end else if (quotient >= 2048) begin
          quotient = quotient >> 1;
          exponent = exponent + 1;
        end
        if (exponent <= 0)
          fp16_div = 16'd0;
        else if (exponent >= 31)
          fp16_div = {sign, 5'h1f, 10'd0};
        else
          fp16_div = {sign, exponent[4:0], quotient[9:0]};
      end
    end
  endfunction

  function [15:0] fp16_exp_lut;
    input [15:0] value;
    begin
      case (value)
        16'hc000: fp16_exp_lut = 16'h3054;
        16'hbc00: fp16_exp_lut = 16'h35e3;
        16'h0000: fp16_exp_lut = 16'h3c00;
        16'h3c00: fp16_exp_lut = 16'h4170;
        16'h4000: fp16_exp_lut = 16'h4764;
        default: fp16_exp_lut = value[15] ? 16'h3800 : 16'h4000;
      endcase
    end
  endfunction

  always @* begin
    case (op_i)
      OP_ADD: result_o = fp16_add(a_i, b_i);
      OP_FMA: result_o = fp16_add(fp16_mul(a_i, b_i), c_i);
      OP_MAX: result_o = fp16_max(a_i, b_i);
      OP_EXP: result_o = fp16_exp_lut(a_i);
      OP_DIV: result_o = fp16_div(a_i, b_i);
      OP_MUL: result_o = fp16_mul(a_i, b_i);
      default: result_o = a_i;
    endcase
  end
endmodule

module mlx_fp16_reduced_lane (
    input  wire [3:0]  op_i,
    input  wire [15:0] a_i,
    input  wire [15:0] b_i,
    input  wire [15:0] c_i,
    output wire [15:0] result_o
);
  wire [3:0] safe_op = ((op_i == 4'd5) || (op_i == 4'd6)) ? 4'd0 : op_i;
  mlx_fp16_alu_lane lane (
      .op_i(safe_op),
      .a_i(a_i),
      .b_i(b_i),
      .c_i(c_i),
      .result_o(result_o)
  );
endmodule

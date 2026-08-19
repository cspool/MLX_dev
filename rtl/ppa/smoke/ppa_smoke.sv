`timescale 1ns/1ps

module ppa_smoke (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_i,
    input  wire [15:0] lhs_i,
    input  wire [15:0] rhs_i,
    input  wire [31:0] addend_i,
    output reg         valid_o,
    output reg  [31:0] result_o,
    output reg  [31:0] checksum_o
);
  reg [31:0] product_q;
  reg [31:0] addend_q;
  reg        valid_q;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      product_q <= 32'd0;
      addend_q <= 32'd0;
      valid_q <= 1'b0;
      valid_o <= 1'b0;
      result_o <= 32'd0;
      checksum_o <= 32'd0;
    end else begin
      valid_q <= valid_i;
      if (valid_i) begin
        product_q <= lhs_i * rhs_i;
        addend_q <= addend_i;
      end
      valid_o <= valid_q;
      if (valid_q) begin
        result_o <= product_q + addend_q;
        checksum_o <= checksum_o ^ (product_q + addend_q);
      end
    end
  end
endmodule

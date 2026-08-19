`timescale 1ns/1ps

module tb_ppa_smoke;
  reg clk = 1'b0;
  reg rst_n = 1'b0;
  reg valid_i = 1'b0;
  reg [15:0] lhs_i = 16'd0;
  reg [15:0] rhs_i = 16'd0;
  reg [31:0] addend_i = 32'd0;
  wire valid_o;
  wire [31:0] result_o;
  wire [31:0] checksum_o;
  integer seen = 0;

  ppa_smoke dut (
      .clk(clk),
      .rst_n(rst_n),
      .valid_i(valid_i),
      .lhs_i(lhs_i),
      .rhs_i(rhs_i),
      .addend_i(addend_i),
      .valid_o(valid_o),
      .result_o(result_o),
      .checksum_o(checksum_o)
  );

  always #0.5 clk = ~clk;

  always @(posedge clk) begin
    if (valid_o) begin
      case (seen)
        0: if (result_o !== 32'd43) $fatal(1, "first result mismatch");
        1: if (result_o !== 32'd1235) $fatal(1, "second result mismatch");
        2: if (result_o !== 32'd262125) $fatal(1, "third result mismatch");
        default: $fatal(1, "unexpected result");
      endcase
      seen = seen + 1;
    end
  end

  initial begin
    $dumpfile("artifacts/environment/h197/ppa-smoke.vcd");
    $dumpvars(0, tb_ppa_smoke);
    repeat (3) @(posedge clk);
    rst_n <= 1'b1;
    @(posedge clk);
    valid_i <= 1'b1;
    lhs_i <= 16'd6;
    rhs_i <= 16'd7;
    addend_i <= 32'd1;
    @(posedge clk);
    lhs_i <= 16'd12;
    rhs_i <= 16'd100;
    addend_i <= 32'd35;
    @(posedge clk);
    lhs_i <= 16'd65535;
    rhs_i <= 16'd2;
    addend_i <= 32'd131055;
    @(posedge clk);
    valid_i <= 1'b0;
    lhs_i <= 16'd0;
    rhs_i <= 16'd0;
    addend_i <= 32'd0;
    repeat (4) @(posedge clk);
    if (seen != 3) $fatal(1, "result count mismatch");
    if (checksum_o !== (32'd43 ^ 32'd1235 ^ 32'd262125))
      $fatal(1, "checksum mismatch");
    $display("PPA_SMOKE_PASS checksum=%0d", checksum_o);
    $finish;
  end
endmodule

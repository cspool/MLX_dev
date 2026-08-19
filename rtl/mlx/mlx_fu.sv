`timescale 1ns/1ps

module mlx_fu #(
    parameter SIMD_WIDTH = 32,
    parameter DATA_BITS = 16,
    parameter FULL_FEATURES = 1,
    parameter TRANS_LANES = (SIMD_WIDTH / 4),
    parameter HP_LANES = 8
) (
    input  wire                            clk,
    input  wire                            rst_n,
    input  wire                            valid_i,
    input  wire [3:0]                      op_i,
    input  wire [SIMD_WIDTH*DATA_BITS-1:0] vector_a_i,
    input  wire [SIMD_WIDTH*DATA_BITS-1:0] vector_b_i,
    input  wire [SIMD_WIDTH*DATA_BITS-1:0] vector_c_i,
    output reg                             valid_o,
    output reg  [SIMD_WIDTH*DATA_BITS-1:0] vector_result_o,
    output reg                             illegal_o,
    output wire [HP_LANES*32-1:0]         high_precision_result_o
);
  localparam OP_EXP = 4'd5;
  localparam OP_DIV = 4'd6;
  localparam OP_SHUFFLE = 4'd7;
  wire [SIMD_WIDTH*DATA_BITS-1:0] lane_result;
  wire removed_operation = (op_i == OP_DIV) || (op_i == OP_SHUFFLE);
  reg [31:0] high_precision_q [0:HP_LANES-1];
  genvar lane;
  genvar hp_lane;

  generate
    for (hp_lane = 0; hp_lane < HP_LANES; hp_lane = hp_lane + 1) begin : GENERATE_HP
      assign high_precision_result_o[hp_lane*32 +: 32] = high_precision_q[hp_lane];
      if (FULL_FEATURES != 0) begin : GENERATE_HP_FULL
        wire [31:0] hp_a = {
            vector_a_i[(2*hp_lane+1)*DATA_BITS +: DATA_BITS],
            vector_a_i[(2*hp_lane)*DATA_BITS +: DATA_BITS]
        };
        wire [31:0] hp_b = {
            vector_b_i[(2*hp_lane+1)*DATA_BITS +: DATA_BITS],
            vector_b_i[(2*hp_lane)*DATA_BITS +: DATA_BITS]
        };
        always @(posedge clk or negedge rst_n) begin
          if (!rst_n)
            high_precision_q[hp_lane] <= 32'd0;
          else if (valid_i)
            high_precision_q[hp_lane] <= hp_a * hp_b + {16'd0, hp_a[15:0]};
        end
      end else begin : GENERATE_HP_REMOVED
        always @* high_precision_q[hp_lane] = 32'd0;
      end
    end
  endgenerate

  generate
    for (lane = 0; lane < SIMD_WIDTH; lane = lane + 1) begin : GENERATE_LANES
      wire [15:0] lane_a = vector_a_i[lane*DATA_BITS +: DATA_BITS];
      wire [15:0] lane_b = vector_b_i[lane*DATA_BITS +: DATA_BITS];
      wire [15:0] lane_c = vector_c_i[lane*DATA_BITS +: DATA_BITS];
      wire [15:0] arithmetic_result;
      if ((FULL_FEATURES != 0) && (lane < TRANS_LANES)) begin : GENERATE_FULL_LANE
        mlx_fp16_alu_lane alu (
            .op_i(op_i),
            .a_i(lane_a),
            .b_i(lane_b),
            .c_i(lane_c),
            .result_o(arithmetic_result)
        );
      end else begin : GENERATE_REDUCED_LANE
        mlx_fp16_reduced_lane alu (
            .op_i(op_i),
            .a_i(lane_a),
            .b_i(lane_b),
            .c_i(lane_c),
            .result_o(arithmetic_result)
        );
      end
      if (FULL_FEATURES != 0) begin : GENERATE_SHUFFLE
        localparam integer INTEGER_SOURCE = lane ^ 1;
        if (lane < TRANS_LANES) begin : GENERATE_TRANS_LANE
          assign lane_result[lane*DATA_BITS +: DATA_BITS] =
              (op_i == OP_SHUFFLE)
                  ? vector_a_i[INTEGER_SOURCE*DATA_BITS +: DATA_BITS]
                  : arithmetic_result;
        end else begin : GENERATE_NON_TRANS_LANE
          assign lane_result[lane*DATA_BITS +: DATA_BITS] =
              (op_i == OP_SHUFFLE)
                  ? vector_a_i[INTEGER_SOURCE*DATA_BITS +: DATA_BITS]
                  : (((op_i == OP_EXP) || (op_i == OP_DIV))
                      ? lane_a : arithmetic_result);
        end
      end else begin : GENERATE_NO_SHUFFLE
        assign lane_result[lane*DATA_BITS +: DATA_BITS] = arithmetic_result;
      end
    end
  endgenerate

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      valid_o <= 1'b0;
      vector_result_o <= {(SIMD_WIDTH*DATA_BITS){1'b0}};
      illegal_o <= 1'b0;
    end else begin
      valid_o <= valid_i;
      illegal_o <= valid_i && (FULL_FEATURES == 0) && removed_operation;
      if (valid_i)
        vector_result_o <= lane_result;
    end
  end
endmodule

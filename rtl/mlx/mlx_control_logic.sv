`timescale 1ns/1ps

module mlx_control_logic #(
    parameter TAGS = 16,
    parameter TAG_BITS = 4,
    parameter STATE_BITS = 64
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire [TAGS-1:0]       active_i,
    input  wire [TAGS-1:0]       ready_i,
    input  wire [2*TAGS-1:0]     pipeline_class_i,
    input  wire [3:0]            pipeline_ready_i,
    output reg  [3:0]            issue_valid_o,
    output reg  [4*TAG_BITS-1:0] issue_tag_o,
    output reg  [15:0]           state_checksum_o,
    output wire [TAGS*STATE_BITS-1:0] schedule_state_o
);
  integer tag_index;
  integer pipeline_index;
  reg selected;
  reg [1:0] tag_pipeline;
  reg [7:0] issue_age_q [0:TAGS-1];
  reg [STATE_BITS-1:0] schedule_state_q [0:TAGS-1];
  integer age_index;
  genvar state_index;

  generate
    for (state_index = 0; state_index < TAGS; state_index = state_index + 1) begin
      assign schedule_state_o[state_index*STATE_BITS +: STATE_BITS]
          = schedule_state_q[state_index];
    end
  endgenerate

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state_checksum_o <= 16'd0;
      for (age_index = 0; age_index < TAGS; age_index = age_index + 1)
        issue_age_q[age_index] <= 8'd0;
      for (age_index = 0; age_index < TAGS; age_index = age_index + 1)
        schedule_state_q[age_index] <= {STATE_BITS{1'b0}};
    end else begin
      state_checksum_o <= 16'd0;
      for (age_index = 0; age_index < TAGS; age_index = age_index + 1) begin
        if (active_i[age_index] && ready_i[age_index])
          issue_age_q[age_index] <= issue_age_q[age_index] + 1'b1;
        if (active_i[age_index]) begin
          schedule_state_q[age_index] <= {
              schedule_state_q[age_index][STATE_BITS-6:0],
              ready_i[age_index],
              pipeline_class_i[2*age_index +: 2],
              issue_age_q[age_index][1:0]
          };
        end
        state_checksum_o <= state_checksum_o
            ^ {8'd0, issue_age_q[age_index]}
            ^ {12'd0, pipeline_class_i[2*age_index +: 2], 2'd0};
      end
    end
  end

  always @* begin
    issue_valid_o = 4'b0000;
    issue_tag_o = {(4*TAG_BITS){1'b0}};
    for (pipeline_index = 0; pipeline_index < 4; pipeline_index = pipeline_index + 1) begin
      selected = 1'b0;
      for (tag_index = 0; tag_index < TAGS; tag_index = tag_index + 1) begin
        tag_pipeline = pipeline_class_i[2*tag_index +: 2];
        if (!selected && active_i[tag_index] && ready_i[tag_index]
            && pipeline_ready_i[pipeline_index]
            && (tag_pipeline == pipeline_index[1:0])) begin
          issue_valid_o[pipeline_index] = 1'b1;
          issue_tag_o[pipeline_index*TAG_BITS +: TAG_BITS] = tag_index[TAG_BITS-1:0];
          selected = 1'b1;
        end
      end
    end
  end
endmodule

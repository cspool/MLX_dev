`timescale 1ns/1ps

module mlx_control_logic #(
    parameter TAGS = 16,
    parameter TAG_BITS = 4
) (
    input  wire [TAGS-1:0]       active_i,
    input  wire [TAGS-1:0]       ready_i,
    input  wire [2*TAGS-1:0]     pipeline_class_i,
    input  wire [3:0]            pipeline_ready_i,
    output reg  [3:0]            issue_valid_o,
    output reg  [4*TAG_BITS-1:0] issue_tag_o
);
  integer tag_index;
  integer pipeline_index;
  reg selected;
  reg [1:0] tag_pipeline;

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

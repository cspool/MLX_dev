`timescale 1ns/1ps

module mlx_tag_buffer #(
    parameter TAGS = 16,
    parameter TAG_BITS = 4,
    parameter METADATA_BITS = 64
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                configure_i,
    input  wire [TAG_BITS-1:0] configure_tag_i,
    input  wire [7:0]          configure_trip_count_i,
    input  wire [5:0]          configure_frontier_i,
    input  wire                configure_ready_i,
    input  wire [METADATA_BITS-1:0] configure_metadata_i,
    input  wire                issue_i,
    input  wire [TAG_BITS-1:0] issue_tag_i,
    input  wire                complete_i,
    input  wire [TAG_BITS-1:0] complete_tag_i,
    input  wire [TAG_BITS-1:0] query_tag_i,
    output wire                query_active_o,
    output wire                query_ready_o,
    output wire                query_done_o,
    output wire [7:0]          query_trip_count_o,
    output wire [5:0]          query_frontier_o,
    output wire [METADATA_BITS-1:0] query_metadata_o,
    output wire [TAGS-1:0]     active_vector_o,
    output wire [TAGS-1:0]     ready_vector_o,
    output wire [TAGS-1:0]     done_vector_o
);
  reg [TAGS-1:0] active_q;
  reg [TAGS-1:0] ready_q;
  reg [TAGS-1:0] done_q;
  reg [7:0] trip_count_q [0:TAGS-1];
  reg [5:0] frontier_q [0:TAGS-1];
  reg [METADATA_BITS-1:0] metadata_q [0:TAGS-1];
  integer index;

  assign query_active_o = active_q[query_tag_i];
  assign query_ready_o = ready_q[query_tag_i];
  assign query_done_o = done_q[query_tag_i];
  assign query_trip_count_o = trip_count_q[query_tag_i];
  assign query_frontier_o = frontier_q[query_tag_i];
  assign query_metadata_o = metadata_q[query_tag_i];
  assign active_vector_o = active_q;
  assign ready_vector_o = ready_q;
  assign done_vector_o = done_q;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      active_q <= {TAGS{1'b0}};
      ready_q <= {TAGS{1'b0}};
      done_q <= {TAGS{1'b0}};
      for (index = 0; index < TAGS; index = index + 1) begin
        trip_count_q[index] <= 8'd0;
        frontier_q[index] <= 6'd0;
        metadata_q[index] <= {METADATA_BITS{1'b0}};
      end
    end else begin
      if (configure_i) begin
        active_q[configure_tag_i] <= 1'b1;
        ready_q[configure_tag_i] <= configure_ready_i;
        done_q[configure_tag_i] <= 1'b0;
        trip_count_q[configure_tag_i] <= configure_trip_count_i;
        frontier_q[configure_tag_i] <= configure_frontier_i;
        metadata_q[configure_tag_i] <= configure_metadata_i;
      end
      if (issue_i && active_q[issue_tag_i] && ready_q[issue_tag_i]) begin
        frontier_q[issue_tag_i] <= frontier_q[issue_tag_i] + 1'b1;
        if (trip_count_q[issue_tag_i] != 0)
          trip_count_q[issue_tag_i] <= trip_count_q[issue_tag_i] - 1'b1;
      end
      if (complete_i) begin
        active_q[complete_tag_i] <= 1'b0;
        ready_q[complete_tag_i] <= 1'b0;
        done_q[complete_tag_i] <= 1'b1;
      end
    end
  end
endmodule

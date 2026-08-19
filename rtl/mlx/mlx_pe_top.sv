`timescale 1ns/1ps

module mlx_pe_top #(
    parameter SIMD_WIDTH = 32,
    parameter FULL_FEATURES = 1,
    parameter TAGS = 16,
    parameter DATA_BITS = 16
) (
    input  wire                            clk,
    input  wire                            rst_n,
    input  wire                            cfg_valid_i,
    input  wire [4:0]                      cfg_addr_i,
    input  wire [63:0]                     cfg_word_i,
    input  wire [4:0]                      fetch_addr_i,
    output wire [63:0]                     fetch_word_o,
    output wire                            configured_o,
    input  wire                            tag_configure_i,
    input  wire [3:0]                      tag_configure_id_i,
    input  wire [7:0]                      tag_trip_count_i,
    input  wire [5:0]                      tag_frontier_i,
    input  wire                            tag_ready_i,
    input  wire                            tag_issue_i,
    input  wire [3:0]                      tag_issue_id_i,
    input  wire                            tag_complete_i,
    input  wire [3:0]                      tag_complete_id_i,
    input  wire [3:0]                      tag_query_id_i,
    output wire [TAGS-1:0]                 tag_active_vector_o,
    output wire [TAGS-1:0]                 tag_ready_vector_o,
    output wire [TAGS-1:0]                 tag_done_vector_o,
    output wire [7:0]                      tag_query_trip_count_o,
    output wire [5:0]                      tag_query_frontier_o,
    input  wire [2*TAGS-1:0]               pipeline_class_i,
    input  wire [3:0]                      pipeline_ready_i,
    output wire [3:0]                      issue_valid_o,
    output wire [15:0]                     issue_tag_o,
    input  wire [3:0]                      rf_read_addr_a_i,
    input  wire [3:0]                      rf_read_addr_b_i,
    output wire [SIMD_WIDTH*DATA_BITS-1:0] rf_read_data_a_o,
    output wire [SIMD_WIDTH*DATA_BITS-1:0] rf_read_data_b_o,
    input  wire                            rf_write_enable_i,
    input  wire [3:0]                      rf_write_addr_i,
    input  wire [SIMD_WIDTH*DATA_BITS-1:0] rf_write_data_i,
    input  wire                            network_valid_i,
    input  wire signed [4:0]               network_dx_i,
    input  wire signed [4:0]               network_dy_i,
    input  wire [3:0]                      network_destination_register_i,
    input  wire [3:0]                      network_tag_i,
    input  wire [63:0]                     network_payload_i,
    output wire                            network_valid_o,
    output wire signed [4:0]               network_dx_o,
    output wire signed [4:0]               network_dy_o,
    output wire [2:0]                      network_route_o,
    output wire [1:0]                      network_consumed_hops_o,
    output wire                            network_delivered_o,
    output wire [63:0]                     network_payload_o,
    input  wire                            fu_valid_i,
    input  wire [3:0]                      fu_op_i,
    input  wire [SIMD_WIDTH*DATA_BITS-1:0] fu_vector_a_i,
    input  wire [SIMD_WIDTH*DATA_BITS-1:0] fu_vector_b_i,
    input  wire [SIMD_WIDTH*DATA_BITS-1:0] fu_vector_c_i,
    output wire                            fu_valid_o,
    output wire [SIMD_WIDTH*DATA_BITS-1:0] fu_vector_result_o,
    output wire                            fu_illegal_o
);
  wire cfg_ready_unused;
  wire query_active_unused;
  wire query_ready_unused;
  wire query_done_unused;
  wire [63:0] query_metadata_unused;
  wire [3:0] network_destination_register_unused;
  wire [3:0] network_tag_unused;
  wire [63:0] network_buffer_observe_unused;
  wire [15:0] control_state_checksum_unused;
  wire [1023:0] control_schedule_state_unused;
  wire [255:0] high_precision_result_unused;

  mlx_config_network config_network (
      .clk(clk),
      .rst_n(rst_n),
      .cfg_valid_i(cfg_valid_i),
      .cfg_addr_i(cfg_addr_i),
      .cfg_word_i(cfg_word_i),
      .cfg_ready_o(cfg_ready_unused),
      .fetch_addr_i(fetch_addr_i),
      .fetch_word_o(fetch_word_o),
      .configured_o(configured_o)
  );

  mlx_tag_buffer tag_buffer (
      .clk(clk),
      .rst_n(rst_n),
      .configure_i(tag_configure_i),
      .configure_tag_i(tag_configure_id_i),
      .configure_trip_count_i(tag_trip_count_i),
      .configure_frontier_i(tag_frontier_i),
      .configure_ready_i(tag_ready_i),
      .configure_metadata_i({32'd0, cfg_word_i[39:8]}),
      .issue_i(tag_issue_i),
      .issue_tag_i(tag_issue_id_i),
      .complete_i(tag_complete_i),
      .complete_tag_i(tag_complete_id_i),
      .query_tag_i(tag_query_id_i),
      .query_active_o(query_active_unused),
      .query_ready_o(query_ready_unused),
      .query_done_o(query_done_unused),
      .query_trip_count_o(tag_query_trip_count_o),
      .query_frontier_o(tag_query_frontier_o),
      .query_metadata_o(query_metadata_unused),
      .active_vector_o(tag_active_vector_o),
      .ready_vector_o(tag_ready_vector_o),
      .done_vector_o(tag_done_vector_o)
  );

  mlx_control_logic control_logic (
      .clk(clk),
      .rst_n(rst_n),
      .active_i(tag_active_vector_o),
      .ready_i(tag_ready_vector_o),
      .pipeline_class_i(pipeline_class_i),
      .pipeline_ready_i(pipeline_ready_i),
      .issue_valid_o(issue_valid_o),
      .issue_tag_o(issue_tag_o),
      .state_checksum_o(control_state_checksum_unused),
      .schedule_state_o(control_schedule_state_unused)
  );

  mlx_register_file #(.SIMD_WIDTH(SIMD_WIDTH)) register_file (
      .clk(clk),
      .rst_n(rst_n),
      .read_addr_a_i(rf_read_addr_a_i),
      .read_addr_b_i(rf_read_addr_b_i),
      .read_data_a_o(rf_read_data_a_o),
      .read_data_b_o(rf_read_data_b_o),
      .write_enable_i(rf_write_enable_i),
      .write_addr_i(rf_write_addr_i),
      .write_data_i(rf_write_data_i)
  );

  mlx_data_network data_network (
      .clk(clk),
      .rst_n(rst_n),
      .valid_i(network_valid_i),
      .dx_i(network_dx_i),
      .dy_i(network_dy_i),
      .destination_register_i(network_destination_register_i),
      .tag_i(network_tag_i),
      .payload_i(network_payload_i),
      .auxiliary_valid_i({4'b0000, network_valid_i}),
      .auxiliary_payload_i({5{network_payload_i}}),
      .ready_o(),
      .valid_o(network_valid_o),
      .dx_o(network_dx_o),
      .dy_o(network_dy_o),
      .destination_register_o(network_destination_register_unused),
      .tag_o(network_tag_unused),
      .payload_o(network_payload_o),
      .route_o(network_route_o),
      .consumed_hops_o(network_consumed_hops_o),
      .delivered_o(network_delivered_o),
      .buffer_observe_o(network_buffer_observe_unused)
  );

  mlx_fu #(
      .SIMD_WIDTH(SIMD_WIDTH),
      .FULL_FEATURES(FULL_FEATURES)
  ) functional_unit (
      .clk(clk),
      .rst_n(rst_n),
      .valid_i(fu_valid_i),
      .op_i(fu_op_i),
      .vector_a_i(fu_vector_a_i),
      .vector_b_i(fu_vector_b_i),
      .vector_c_i(fu_vector_c_i),
      .valid_o(fu_valid_o),
      .vector_result_o(fu_vector_result_o),
      .illegal_o(fu_illegal_o),
      .high_precision_result_o(high_precision_result_unused)
  );
endmodule

`ifndef MLX_NO_WRAPPERS
module mlx_pe_full (
    input wire clk,
    input wire rst_n,
    input wire cfg_valid_i,
    input wire [4:0] cfg_addr_i,
    input wire [63:0] cfg_word_i,
    output wire [63:0] fetch_word_o,
    input wire fu_valid_i,
    input wire [3:0] fu_op_i,
    input wire [511:0] fu_vector_a_i,
    input wire [511:0] fu_vector_b_i,
    input wire [511:0] fu_vector_c_i,
    output wire fu_valid_o,
    output wire [511:0] fu_vector_result_o,
    output wire fu_illegal_o,
    input wire network_valid_i,
    input wire signed [4:0] network_dx_i,
    input wire signed [4:0] network_dy_i,
    output wire network_valid_o,
    output wire signed [4:0] network_dx_o,
    output wire signed [4:0] network_dy_o,
    output wire [2:0] network_route_o,
    output wire [1:0] network_consumed_hops_o
);
  wire [15:0] tag_active;
  wire [15:0] tag_ready;
  wire [15:0] tag_done;
  wire [511:0] rf_a;
  wire [511:0] rf_b;
  wire [3:0] issue_valid;
  wire [15:0] issue_tag;
  mlx_pe_top #(.SIMD_WIDTH(32), .FULL_FEATURES(1)) pe (
      .clk(clk), .rst_n(rst_n),
      .cfg_valid_i(cfg_valid_i), .cfg_addr_i(cfg_addr_i), .cfg_word_i(cfg_word_i),
      .fetch_addr_i(cfg_addr_i), .fetch_word_o(fetch_word_o), .configured_o(),
      .tag_configure_i(cfg_valid_i), .tag_configure_id_i(cfg_addr_i[3:0]),
      .tag_trip_count_i(cfg_word_i[7:0]), .tag_frontier_i(6'd0), .tag_ready_i(1'b1),
      .tag_issue_i(1'b0), .tag_issue_id_i(4'd0), .tag_complete_i(1'b0),
      .tag_complete_id_i(4'd0), .tag_query_id_i(4'd0),
      .tag_active_vector_o(tag_active), .tag_ready_vector_o(tag_ready),
      .tag_done_vector_o(tag_done), .tag_query_trip_count_o(), .tag_query_frontier_o(),
      .pipeline_class_i(32'd0), .pipeline_ready_i(4'hf),
      .issue_valid_o(issue_valid), .issue_tag_o(issue_tag),
      .rf_read_addr_a_i(4'd0), .rf_read_addr_b_i(4'd1),
      .rf_read_data_a_o(rf_a), .rf_read_data_b_o(rf_b),
      .rf_write_enable_i(cfg_valid_i), .rf_write_addr_i(cfg_addr_i[3:0]),
      .rf_write_data_i({8{cfg_word_i}}),
      .network_valid_i(network_valid_i), .network_dx_i(network_dx_i),
      .network_dy_i(network_dy_i), .network_destination_register_i(4'd0),
      .network_tag_i(4'd0), .network_payload_i(cfg_word_i),
      .network_valid_o(network_valid_o), .network_dx_o(network_dx_o),
      .network_dy_o(network_dy_o), .network_route_o(network_route_o),
      .network_consumed_hops_o(network_consumed_hops_o), .network_delivered_o(),
      .network_payload_o(), .fu_valid_i(fu_valid_i), .fu_op_i(fu_op_i),
      .fu_vector_a_i(fu_vector_a_i), .fu_vector_b_i(fu_vector_b_i),
      .fu_vector_c_i(fu_vector_c_i), .fu_valid_o(fu_valid_o),
      .fu_vector_result_o(fu_vector_result_o), .fu_illegal_o(fu_illegal_o)
  );
endmodule

module mlx_pe_reduced (
    input wire clk,
    input wire rst_n,
    input wire cfg_valid_i,
    input wire [4:0] cfg_addr_i,
    input wire [63:0] cfg_word_i,
    output wire [63:0] fetch_word_o,
    input wire fu_valid_i,
    input wire [3:0] fu_op_i,
    input wire [127:0] fu_vector_a_i,
    input wire [127:0] fu_vector_b_i,
    input wire [127:0] fu_vector_c_i,
    output wire fu_valid_o,
    output wire [127:0] fu_vector_result_o,
    output wire fu_illegal_o
);
  mlx_pe_top #(.SIMD_WIDTH(8), .FULL_FEATURES(0)) pe (
      .clk(clk), .rst_n(rst_n),
      .cfg_valid_i(cfg_valid_i), .cfg_addr_i(cfg_addr_i), .cfg_word_i(cfg_word_i),
      .fetch_addr_i(cfg_addr_i), .fetch_word_o(fetch_word_o), .configured_o(),
      .tag_configure_i(cfg_valid_i), .tag_configure_id_i(cfg_addr_i[3:0]),
      .tag_trip_count_i(cfg_word_i[7:0]), .tag_frontier_i(6'd0), .tag_ready_i(1'b1),
      .tag_issue_i(1'b0), .tag_issue_id_i(4'd0), .tag_complete_i(1'b0),
      .tag_complete_id_i(4'd0), .tag_query_id_i(4'd0),
      .tag_active_vector_o(), .tag_ready_vector_o(), .tag_done_vector_o(),
      .tag_query_trip_count_o(), .tag_query_frontier_o(),
      .pipeline_class_i(32'd0), .pipeline_ready_i(4'hf),
      .issue_valid_o(), .issue_tag_o(),
      .rf_read_addr_a_i(4'd0), .rf_read_addr_b_i(4'd1),
      .rf_read_data_a_o(), .rf_read_data_b_o(),
      .rf_write_enable_i(cfg_valid_i), .rf_write_addr_i(cfg_addr_i[3:0]),
      .rf_write_data_i({2{cfg_word_i}}),
      .network_valid_i(1'b0), .network_dx_i(5'sd0), .network_dy_i(5'sd0),
      .network_destination_register_i(4'd0), .network_tag_i(4'd0),
      .network_payload_i(64'd0), .network_valid_o(), .network_dx_o(),
      .network_dy_o(), .network_route_o(), .network_consumed_hops_o(),
      .network_delivered_o(), .network_payload_o(),
      .fu_valid_i(fu_valid_i), .fu_op_i(fu_op_i),
      .fu_vector_a_i(fu_vector_a_i), .fu_vector_b_i(fu_vector_b_i),
      .fu_vector_c_i(fu_vector_c_i), .fu_valid_o(fu_valid_o),
      .fu_vector_result_o(fu_vector_result_o), .fu_illegal_o(fu_illegal_o)
  );
endmodule
`endif

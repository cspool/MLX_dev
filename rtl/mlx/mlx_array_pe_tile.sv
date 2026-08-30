`timescale 1ns/1ps

// Autonomous PE tile used by the physically distributed array hierarchy.
//
// The legacy array controller drove the RF and FU vector ports of every hard
// PE from the array top.  That made logically local 512-bit paths cross the PE
// boundary and synthesized the packet delivery path as a global crossbar.  A
// tile owns those paths instead.  The array top sees only configuration, one
// shared-SPM request, and one registered packet input/output per tile.
module mlx_array_pe_tile #(
    parameter SIMD_WIDTH = 32,
    parameter DATA_BITS = 16,
    parameter VECTOR_BITS = SIMD_WIDTH * DATA_BITS,
    parameter TRANS_LANES = (SIMD_WIDTH / 4),
    parameter PROGRAM_DEPTH = 32
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   cfg_valid_i,
    input  wire [4:0]             cfg_addr_i,
    input  wire [63:0]            cfg_word_i,
    input  wire                   cfg_instruction_count_valid_i,
    input  wire [5:0]             cfg_instruction_count_i,
    input  wire                   launch_i,
    input  wire [3:0]             tile_id_i,

    output wire                   spm_req_valid_o,
    output wire                   spm_req_write_o,
    output wire [7:0]             spm_req_addr_o,
    output wire [VECTOR_BITS-1:0] spm_req_wdata_o,
    input  wire                   spm_req_grant_i,
    input  wire                   spm_rsp_valid_i,
    input  wire [VECTOR_BITS-1:0] spm_rsp_rdata_i,

    input  wire                   route_in_valid_i,
    input  wire signed [4:0]      route_in_dx_i,
    input  wire signed [4:0]      route_in_dy_i,
    input  wire [3:0]             route_in_destination_register_i,
    input  wire [3:0]             route_in_tag_i,
    input  wire [3:0]             route_in_source_i,
    input  wire [VECTOR_BITS-1:0] route_in_data_i,
    output wire                   route_in_ready_o,

    output wire                   route_out_valid_o,
    output reg  [4:0]             route_out_target_o,
    output reg  signed [4:0]      route_out_dx_o,
    output reg  signed [4:0]      route_out_dy_o,
    output wire [3:0]             route_out_destination_register_o,
    output wire [3:0]             route_out_tag_o,
    output wire [3:0]             route_out_source_o,
    output wire [VECTOR_BITS-1:0] route_out_data_o,
    output reg  [1:0]             route_out_hops_o,
    input  wire                   route_out_grant_i,

    input  wire                   xfer_complete_i,
    output wire                   xfer_complete_valid_o,
    output wire [3:0]             xfer_complete_source_o,

    output wire                   execution_active_o,
    output wire                   router_active_o,
    output wire                   instruction_issue_o,
    output wire                   load_issue_o,
    output wire                   store_issue_o,
    output wire                   compute_issue_o,
    output wire                   xfer_issue_o,
    output wire                   stall_o,
    output wire                   delivery_conflict_o
);
  localparam ST_IDLE = 3'd0;
  localparam ST_READY = 3'd1;
  localparam ST_COMPUTE = 3'd2;
  localparam ST_LOAD = 3'd3;
  localparam ST_XFER = 3'd4;
  localparam ST_DONE = 3'd5;

  localparam OP_LOAD = 4'd0;
  localparam OP_STORE = 4'd1;
  localparam OP_FMA = 4'd2;
  localparam OP_ADD = 4'd3;
  localparam OP_MAX = 4'd4;
  localparam OP_EXP = 4'd5;
  localparam OP_DIV = 4'd6;
  localparam OP_SHUFFLE = 4'd7;
  localparam OP_XFER = 4'd8;
  localparam OP_MUL = 4'd9;

  reg [5:0] instruction_count_q;
  reg [5:0] pc_q;
  reg [2:0] state_q;
  reg [5:0] compute_countdown_q;
  reg [3:0] pending_dst_q;
  reg [15:0] rf_valid_q;
  reg tag_complete_q;
  reg xfer_injected_q;
  reg [63:0] debug_cycle_q;

  reg router_valid_q;
  reg signed [4:0] router_dx_q;
  reg signed [4:0] router_dy_q;
  reg [3:0] router_destination_register_q;
  reg [3:0] router_tag_q;
  reg [3:0] router_source_q;
  reg [VECTOR_BITS-1:0] router_data_q;

  wire [63:0] instruction_word;
  wire [3:0] decoded_operation = instruction_word[63:60];
  wire [3:0] decoded_tag = instruction_word[59:56];
  wire [1:0] decoded_pipeline = instruction_word[55:54];
  wire [3:0] decoded_destination = instruction_word[53:50];
  wire [3:0] decoded_source_a = instruction_word[49:46];
  wire [3:0] decoded_source_b = instruction_word[45:42];
  wire [3:0] decoded_source_c = instruction_word[41:38];
  wire signed [4:0] decoded_dx = instruction_word[37:33];
  wire signed [4:0] decoded_dy = instruction_word[32:28];
  wire [7:0] decoded_spm_address = instruction_word[27:20];

  wire [VECTOR_BITS-1:0] rf_a;
  wire [VECTOR_BITS-1:0] rf_b;
  wire [VECTOR_BITS-1:0] rf_c;
  wire [VECTOR_BITS-1:0] fu_result;
  wire fu_result_valid;
  wire fu_illegal;
  wire [3:0] control_issue_valid;
  wire [15:0] control_issue_tag;
  wire [15:0] tag_active;
  wire [15:0] tag_ready;
  wire [15:0] tag_done;

  reg [31:0] pipeline_class;
  reg rf_write_enable;
  reg [3:0] rf_write_addr;
  reg [VECTOR_BITS-1:0] rf_write_data;
  reg fu_valid;

  wire network_valid_unused;
  wire signed [4:0] network_dx_unused;
  wire signed [4:0] network_dy_unused;
  wire [2:0] network_route_unused;
  wire [1:0] network_hops_unused;
  wire network_delivered_unused;
  wire [63:0] network_payload_unused;

  wire operands_are_ready;
  wire compute_commit;
  wire local_write_reserved;
  wire packet_at_destination;
  wire packet_delivery_accept;
  wire local_injection_accept;

  function [5:0] operation_latency;
    input [3:0] operation;
    begin
      case (operation)
        OP_FMA: operation_latency = 6'd4;
        OP_ADD: operation_latency = 6'd3;
        OP_MAX: operation_latency = 6'd1;
        OP_EXP: operation_latency = 6'd8;
        OP_DIV: operation_latency = 6'd12;
        OP_SHUFFLE: operation_latency = 6'd1;
        OP_MUL: operation_latency = 6'd3;
        default: operation_latency = 6'd1;
      endcase
    end
  endfunction

  function sources_ready;
    input [3:0] operation;
    input [15:0] valid_vector;
    input [3:0] source0;
    input [3:0] source1;
    input [3:0] source2;
    begin
      case (operation)
        OP_LOAD: sources_ready = 1'b1;
        OP_STORE, OP_EXP, OP_SHUFFLE, OP_XFER:
          sources_ready = valid_vector[source0];
        OP_FMA:
          sources_ready = valid_vector[source0]
              && valid_vector[source1] && valid_vector[source2];
        OP_ADD, OP_MAX, OP_DIV, OP_MUL:
          sources_ready = valid_vector[source0] && valid_vector[source1];
        default: sources_ready = 1'b0;
      endcase
    end
  endfunction

  assign operands_are_ready = sources_ready(
      decoded_operation,
      rf_valid_q,
      decoded_source_a,
      decoded_source_b,
      decoded_source_c
  );
  assign compute_commit = (state_q == ST_COMPUTE) && (compute_countdown_q == 1);
  assign local_write_reserved = compute_commit || spm_rsp_valid_i;

  assign compute_issue_o = (state_q == ST_READY) && operands_are_ready
      && (decoded_operation >= OP_FMA) && (decoded_operation <= OP_MUL)
      && (decoded_operation != OP_XFER)
      && control_issue_valid[decoded_pipeline];
  assign xfer_issue_o = (state_q == ST_READY) && operands_are_ready
      && (decoded_operation == OP_XFER) && control_issue_valid[3];
  assign spm_req_valid_o = (state_q == ST_READY) && operands_are_ready
      && ((decoded_operation == OP_LOAD) || (decoded_operation == OP_STORE))
      && control_issue_valid[decoded_pipeline];
  assign spm_req_write_o = decoded_operation == OP_STORE;
  assign spm_req_addr_o = decoded_spm_address;
  assign spm_req_wdata_o = rf_a;
  assign load_issue_o = spm_req_valid_o && spm_req_grant_i && !spm_req_write_o;
  assign store_issue_o = spm_req_valid_o && spm_req_grant_i && spm_req_write_o;
  assign instruction_issue_o = compute_issue_o || xfer_issue_o
      || load_issue_o || store_issue_o;
  assign stall_o = (state_q == ST_READY) && !instruction_issue_o;

  assign packet_at_destination = router_valid_q
      && (router_dx_q == 0) && (router_dy_q == 0);
  assign packet_delivery_accept = packet_at_destination && !local_write_reserved;
  assign delivery_conflict_o = packet_at_destination && local_write_reserved;
  assign route_in_ready_o = !router_valid_q;
  assign local_injection_accept = (state_q == ST_XFER) && !xfer_injected_q
      && !router_valid_q && !route_in_valid_i;

  assign route_out_valid_o = router_valid_q && !packet_at_destination;
  assign route_out_destination_register_o = router_destination_register_q;
  assign route_out_tag_o = router_tag_q;
  assign route_out_source_o = router_source_q;
  assign route_out_data_o = router_data_q;
  assign xfer_complete_valid_o = packet_delivery_accept;
  assign xfer_complete_source_o = router_source_q;
  assign execution_active_o = (state_q != ST_IDLE) && (state_q != ST_DONE);
  assign router_active_o = router_valid_q;

  always @* begin
    route_out_target_o = {1'b0, tile_id_i};
    route_out_dx_o = router_dx_q;
    route_out_dy_o = router_dy_q;
    route_out_hops_o = 2'd0;
    if (router_dx_q > 0) begin
      route_out_hops_o = (router_dx_q >= 2) ? 2'd2 : 2'd1;
      route_out_dx_o = router_dx_q - $signed({3'd0, route_out_hops_o});
      route_out_target_o = {1'b0, tile_id_i} + route_out_hops_o;
    end else if (router_dx_q < 0) begin
      route_out_hops_o = ((-router_dx_q) >= 2) ? 2'd2 : 2'd1;
      route_out_dx_o = router_dx_q + $signed({3'd0, route_out_hops_o});
      route_out_target_o = {1'b0, tile_id_i} - route_out_hops_o;
    end else if (router_dy_q > 0) begin
      route_out_hops_o = (router_dy_q >= 2) ? 2'd2 : 2'd1;
      route_out_dy_o = router_dy_q - $signed({3'd0, route_out_hops_o});
      route_out_target_o = {1'b0, tile_id_i}
          + ({3'd0, route_out_hops_o} << 2);
    end else if (router_dy_q < 0) begin
      route_out_hops_o = ((-router_dy_q) >= 2) ? 2'd2 : 2'd1;
      route_out_dy_o = router_dy_q + $signed({3'd0, route_out_hops_o});
      route_out_target_o = {1'b0, tile_id_i}
          - ({3'd0, route_out_hops_o} << 2);
    end
  end

  always @* begin
    pipeline_class = 32'd0;
    pipeline_class[2*tile_id_i +: 2] = decoded_pipeline;
    fu_valid = compute_issue_o;

    rf_write_enable = 1'b0;
    rf_write_addr = 4'd0;
    rf_write_data = {VECTOR_BITS{1'b0}};
    if (compute_commit) begin
      rf_write_enable = 1'b1;
      rf_write_addr = pending_dst_q;
      rf_write_data = fu_result;
    end
    if (spm_rsp_valid_i) begin
      rf_write_enable = 1'b1;
      rf_write_addr = pending_dst_q;
      rf_write_data = spm_rsp_rdata_i;
    end
    if (packet_delivery_accept) begin
      rf_write_enable = 1'b1;
      rf_write_addr = router_destination_register_q;
      rf_write_data = router_data_q;
    end
  end

`ifdef MLX_PPA_MACRO
  mlx_pe_top physical_pe (
`else
  mlx_pe_top #(
      .SIMD_WIDTH(SIMD_WIDTH),
      .FULL_FEATURES(1),
      .TRANS_LANES(TRANS_LANES),
      .RF_DEPTH(16),
      .CONFIG_GATED_CLOCK(0),
      .TAG_GATED_CLOCK(0),
      .RF_GATED_CLOCK(0),
      .NETWORK_GATED_CLOCK(0)
  ) physical_pe (
`endif
      .clk(clk),
      .rst_n(rst_n),
      .cfg_valid_i(cfg_valid_i && (cfg_addr_i < PROGRAM_DEPTH)),
      .cfg_addr_i(cfg_addr_i),
      .cfg_word_i(cfg_word_i),
      .fetch_addr_i(pc_q[4:0]),
      .fetch_word_o(instruction_word),
      .configured_o(),
      .tag_configure_i(launch_i && (instruction_count_q != 0)),
      .tag_configure_id_i(tile_id_i),
      .tag_trip_count_i({2'd0, instruction_count_q}),
      .tag_frontier_i(6'd0),
      .tag_ready_i(1'b1),
      .tag_issue_i(instruction_issue_o),
      .tag_issue_id_i(decoded_tag),
      .tag_complete_i(tag_complete_q),
      .tag_complete_id_i(tile_id_i),
      .tag_query_id_i(tile_id_i),
      .tag_active_vector_o(tag_active),
      .tag_ready_vector_o(tag_ready),
      .tag_done_vector_o(tag_done),
      .tag_query_trip_count_o(),
      .tag_query_frontier_o(),
      .pipeline_class_i(pipeline_class),
      .pipeline_ready_i(4'hf),
      .issue_valid_o(control_issue_valid),
      .issue_tag_o(control_issue_tag),
      .rf_read_addr_a_i(decoded_source_a),
      .rf_read_addr_b_i(decoded_source_b),
      .rf_read_addr_c_i(decoded_source_c),
      .rf_read_data_a_o(rf_a),
      .rf_read_data_b_o(rf_b),
      .rf_read_data_c_o(rf_c),
      .rf_write_enable_i(rf_write_enable),
      .rf_write_addr_i(rf_write_addr),
      .rf_write_data_i(rf_write_data),
      .network_valid_i(router_valid_q),
      .network_dx_i(router_dx_q),
      .network_dy_i(router_dy_q),
      .network_destination_register_i(router_destination_register_q),
      .network_tag_i(router_tag_q),
      .network_payload_i(router_data_q[63:0]),
      .network_valid_o(network_valid_unused),
      .network_dx_o(network_dx_unused),
      .network_dy_o(network_dy_unused),
      .network_route_o(network_route_unused),
      .network_consumed_hops_o(network_hops_unused),
      .network_delivered_o(network_delivered_unused),
      .network_payload_o(network_payload_unused),
      .fu_valid_i(fu_valid),
      .fu_op_i(decoded_operation),
      .fu_vector_a_i(rf_a),
      .fu_vector_b_i(rf_b),
      .fu_vector_c_i(rf_c),
      .fu_valid_o(fu_result_valid),
      .fu_vector_result_o(fu_result),
      .fu_illegal_o(fu_illegal)
  );

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      instruction_count_q <= 6'd0;
      pc_q <= 6'd0;
      state_q <= ST_IDLE;
      compute_countdown_q <= 6'd0;
      pending_dst_q <= 4'd0;
      rf_valid_q <= 16'd0;
      tag_complete_q <= 1'b0;
      xfer_injected_q <= 1'b0;
      debug_cycle_q <= 64'd0;
      router_valid_q <= 1'b0;
      router_dx_q <= 5'sd0;
      router_dy_q <= 5'sd0;
      router_destination_register_q <= 4'd0;
      router_tag_q <= 4'd0;
      router_source_q <= 4'd0;
      router_data_q <= {VECTOR_BITS{1'b0}};
    end else begin
      tag_complete_q <= 1'b0;

      if (cfg_instruction_count_valid_i)
        instruction_count_q <= cfg_instruction_count_i;

      if (launch_i) begin
        debug_cycle_q <= 64'd0;
        pc_q <= 6'd0;
        state_q <= (instruction_count_q == 0) ? ST_DONE : ST_READY;
        compute_countdown_q <= 6'd0;
        pending_dst_q <= 4'd0;
        rf_valid_q <= 16'd0;
        xfer_injected_q <= 1'b0;
        router_valid_q <= 1'b0;
        router_dx_q <= 5'sd0;
        router_dy_q <= 5'sd0;
      end else begin
        debug_cycle_q <= debug_cycle_q + 1'b1;
        if (compute_issue_o) begin
          state_q <= ST_COMPUTE;
          compute_countdown_q <= operation_latency(decoded_operation);
          pending_dst_q <= decoded_destination;
          rf_valid_q[decoded_destination] <= 1'b0;
`ifndef SYNTHESIS
          $display("MLX_TRACE backend=rtl event=issue cycle=%0d pe=%0d pc=%0d op=%0d tag=%0d",
                   debug_cycle_q, tile_id_i, pc_q, decoded_operation, decoded_tag);
`endif
        end else if (state_q == ST_COMPUTE) begin
          if (compute_countdown_q > 1) begin
            compute_countdown_q <= compute_countdown_q - 1'b1;
          end else begin
            rf_valid_q[pending_dst_q] <= 1'b1;
`ifndef SYNTHESIS
            $display("MLX_TRACE backend=rtl event=complete cycle=%0d pe=%0d pc=%0d op=%0d",
                     debug_cycle_q, tile_id_i, pc_q, decoded_operation);
`endif
            if (pc_q + 1 >= instruction_count_q) begin
              state_q <= ST_DONE;
              tag_complete_q <= 1'b1;
            end else begin
              pc_q <= pc_q + 1'b1;
              state_q <= ST_READY;
            end
          end
        end

        if (xfer_issue_o) begin
          state_q <= ST_XFER;
          xfer_injected_q <= 1'b0;
`ifndef SYNTHESIS
          $display("MLX_TRACE backend=rtl event=issue cycle=%0d pe=%0d pc=%0d op=8 tag=%0d",
                   debug_cycle_q, tile_id_i, pc_q, decoded_tag);
`endif
        end

        if (load_issue_o) begin
          state_q <= ST_LOAD;
          pending_dst_q <= decoded_destination;
          rf_valid_q[decoded_destination] <= 1'b0;
`ifndef SYNTHESIS
          $display("MLX_TRACE backend=rtl event=issue cycle=%0d pe=%0d pc=%0d op=0 tag=%0d",
                   debug_cycle_q, tile_id_i, pc_q, decoded_tag);
`endif
        end else if (store_issue_o) begin
`ifndef SYNTHESIS
          $display("MLX_TRACE backend=rtl event=issue cycle=%0d pe=%0d pc=%0d op=1 tag=%0d",
                   debug_cycle_q, tile_id_i, pc_q, decoded_tag);
          $display("MLX_TRACE backend=rtl event=complete cycle=%0d pe=%0d pc=%0d op=1",
                   debug_cycle_q + 1'b1, tile_id_i, pc_q);
`endif
          if (pc_q + 1 >= instruction_count_q) begin
            state_q <= ST_DONE;
            tag_complete_q <= 1'b1;
          end else begin
            pc_q <= pc_q + 1'b1;
            state_q <= ST_READY;
          end
        end

        if (spm_rsp_valid_i) begin
          rf_valid_q[pending_dst_q] <= 1'b1;
`ifndef SYNTHESIS
          $display("MLX_TRACE backend=rtl event=complete cycle=%0d pe=%0d pc=%0d op=0",
                   debug_cycle_q, tile_id_i, pc_q);
`endif
          if (pc_q + 1 >= instruction_count_q) begin
            state_q <= ST_DONE;
            tag_complete_q <= 1'b1;
          end else begin
            pc_q <= pc_q + 1'b1;
            state_q <= ST_READY;
          end
        end

        if (xfer_complete_i && (state_q == ST_XFER)) begin
          xfer_injected_q <= 1'b0;
`ifndef SYNTHESIS
          $display("MLX_TRACE backend=rtl event=complete cycle=%0d pe=%0d pc=%0d op=8",
                   debug_cycle_q, tile_id_i, pc_q);
`endif
          if (pc_q + 1 >= instruction_count_q) begin
            state_q <= ST_DONE;
            tag_complete_q <= 1'b1;
          end else begin
            pc_q <= pc_q + 1'b1;
            state_q <= ST_READY;
          end
        end

        if (packet_delivery_accept) begin
          router_valid_q <= 1'b0;
          rf_valid_q[router_destination_register_q] <= 1'b1;
        end else if (route_out_grant_i) begin
          router_valid_q <= 1'b0;
        end

        if (route_in_valid_i && route_in_ready_o) begin
          router_valid_q <= 1'b1;
          router_dx_q <= route_in_dx_i;
          router_dy_q <= route_in_dy_i;
          router_destination_register_q <= route_in_destination_register_i;
          router_tag_q <= route_in_tag_i;
          router_source_q <= route_in_source_i;
          router_data_q <= route_in_data_i;
        end else if (local_injection_accept) begin
          router_valid_q <= 1'b1;
          router_dx_q <= decoded_dx;
          router_dy_q <= decoded_dy;
          router_destination_register_q <= decoded_destination;
          router_tag_q <= decoded_tag;
          router_source_q <= tile_id_i;
          router_data_q <= rf_a;
          xfer_injected_q <= 1'b1;
        end
      end
    end
  end
endmodule

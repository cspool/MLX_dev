`timescale 1ns/1ps

// Autonomous, executable 4x4 MLX array.  Each physical PE owns its program,
// tag state, SIMD32 RF/FU, and router.  A single vector SPM port is shared by
// deterministic lower-coordinate arbitration; packets advance by one or two
// mesh coordinates per cycle and hold on destination write-port conflicts.
module mlx_array_4x4 #(
    parameter SIMD_WIDTH = 32,
    parameter DATA_BITS = 16,
    parameter VECTOR_BITS = SIMD_WIDTH * DATA_BITS,
    parameter TRANS_LANES = (SIMD_WIDTH / 4),
    parameter PE_COUNT = 16,
    parameter PROGRAM_DEPTH = 32,
    parameter SPM_VECTORS = 128
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   cfg_valid_i,
    input  wire [4:0]             cfg_pe_i,
    input  wire [5:0]             cfg_index_i,
    input  wire [63:0]            cfg_word_i,
    input  wire                   launch_i,
    input  wire [7:0]             input_vectors_i,
    output wire                   spm_req_valid_o,
    input  wire                   spm_req_ready_i,
    output wire                   spm_req_write_o,
    output wire [7:0]             spm_req_addr_o,
    output wire [VECTOR_BITS-1:0] spm_req_wdata_o,
    input  wire                   spm_rsp_valid_i,
    input  wire [VECTOR_BITS-1:0] spm_rsp_rdata_i,
    output wire                   busy_o,
    output reg                    done_o,
    output wire [63:0]            stat_cycles_o,
    output wire [63:0]            stat_instructions_o,
    output wire [63:0]            stat_load_o,
    output wire [63:0]            stat_store_o,
    output wire [63:0]            stat_compute_o,
    output wire [63:0]            stat_xfer_o,
    output wire [63:0]            stat_stall_o,
    output wire [63:0]            stat_hops_o,
    output wire [63:0]            stat_conflicts_o
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

  reg [5:0] instruction_count_q [0:PE_COUNT-1];
  reg [5:0] pc_q [0:PE_COUNT-1];
  reg [2:0] state_q [0:PE_COUNT-1];
  reg [5:0] compute_countdown_q [0:PE_COUNT-1];
  reg [3:0] pending_dst_q [0:PE_COUNT-1];
  reg [15:0] rf_valid_q [0:PE_COUNT-1];
  reg [SPM_VECTORS-1:0] spm_valid_q;
  reg running_q;

  reg spm_pending_q;
  reg [3:0] spm_pending_pe_q;
  reg [3:0] spm_pending_dst_q;

  reg packet_active_q [0:PE_COUNT-1];
  reg [4:0] packet_current_q [0:PE_COUNT-1];
  reg signed [4:0] packet_dx_q [0:PE_COUNT-1];
  reg signed [4:0] packet_dy_q [0:PE_COUNT-1];
  reg [3:0] packet_dst_q [0:PE_COUNT-1];
  reg [3:0] packet_tag_q [0:PE_COUNT-1];
  reg [VECTOR_BITS-1:0] packet_data_q [0:PE_COUNT-1];

  reg [63:0] stat_cycles_q;
  reg [63:0] stat_instructions_q;
  reg [63:0] stat_load_q;
  reg [63:0] stat_store_q;
  reg [63:0] stat_compute_q;
  reg [63:0] stat_xfer_q;
  reg [63:0] stat_stall_q;
  reg [63:0] stat_hops_q;
  reg [63:0] stat_conflicts_q;

  wire [63:0] instruction_word [0:PE_COUNT-1];
  wire [VECTOR_BITS-1:0] rf_a [0:PE_COUNT-1];
  wire [VECTOR_BITS-1:0] rf_b [0:PE_COUNT-1];
  wire [VECTOR_BITS-1:0] rf_c [0:PE_COUNT-1];
  wire [VECTOR_BITS-1:0] fu_result [0:PE_COUNT-1];
  wire fu_result_valid [0:PE_COUNT-1];
  wire fu_illegal [0:PE_COUNT-1];
  wire [3:0] control_issue_valid [0:PE_COUNT-1];
  wire [15:0] control_issue_tag [0:PE_COUNT-1];
  wire [15:0] tag_active [0:PE_COUNT-1];
  wire [15:0] tag_ready [0:PE_COUNT-1];
  wire [15:0] tag_done [0:PE_COUNT-1];

  reg [3:0] rf_read_a [0:PE_COUNT-1];
  reg [3:0] rf_read_b [0:PE_COUNT-1];
  reg [3:0] rf_read_c [0:PE_COUNT-1];
  reg rf_write_enable [0:PE_COUNT-1];
  reg [3:0] rf_write_addr [0:PE_COUNT-1];
  reg [VECTOR_BITS-1:0] rf_write_data [0:PE_COUNT-1];
  reg fu_valid [0:PE_COUNT-1];
  reg [3:0] fu_op [0:PE_COUNT-1];
  reg [31:0] pipeline_class [0:PE_COUNT-1];
  reg tag_complete_q [0:PE_COUNT-1];
  reg network_valid [0:PE_COUNT-1];
  reg signed [4:0] network_dx [0:PE_COUNT-1];
  reg signed [4:0] network_dy [0:PE_COUNT-1];
  reg [3:0] network_dst [0:PE_COUNT-1];
  reg [3:0] network_tag [0:PE_COUNT-1];
  reg [63:0] network_payload [0:PE_COUNT-1];

  reg spm_select_valid;
  reg [3:0] spm_select_pe;
  reg spm_select_write;
  reg [7:0] spm_select_addr;
  reg [VECTOR_BITS-1:0] spm_select_wdata;
  reg issue_any [0:PE_COUNT-1];
  reg issue_xfer [0:PE_COUNT-1];
  reg operands_are_ready [0:PE_COUNT-1];
  reg compute_commit [0:PE_COUNT-1];
  reg packet_route_grant [0:PE_COUNT-1];
  reg packet_will_deliver [0:PE_COUNT-1];
  reg packet_delivery_accept [0:PE_COUNT-1];
  reg [4:0] packet_next_current [0:PE_COUNT-1];
  reg signed [4:0] packet_next_dx [0:PE_COUNT-1];
  reg signed [4:0] packet_next_dy [0:PE_COUNT-1];
  reg [1:0] packet_step [0:PE_COUNT-1];
  reg backend_has_work;
  reg [63:0] cycle_instruction_issues;
  reg [63:0] cycle_load_issues;
  reg [63:0] cycle_store_issues;
  reg [63:0] cycle_compute_issues;
  reg [63:0] cycle_xfer_issues;
  reg [63:0] cycle_stalls;
  reg [63:0] cycle_hops;
  reg [63:0] cycle_conflicts;

  integer comb_pe;
  integer comb_other;
  integer comb_router;
  integer sequential_pe;
  integer reset_index;

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

  genvar pe;
  generate
    for (pe = 0; pe < PE_COUNT; pe = pe + 1) begin : GENERATE_PES
      wire [3:0] decoded_pipeline = instruction_word[pe][55:54];
      wire [3:0] decoded_tag = instruction_word[pe][59:56];
      wire network_valid_unused;
      wire signed [4:0] network_dx_unused;
      wire signed [4:0] network_dy_unused;
      wire [2:0] network_route_unused;
      wire [1:0] network_hops_unused;
      wire network_delivered_unused;
      wire [63:0] network_payload_unused;
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
          .cfg_valid_i(cfg_valid_i && (cfg_pe_i == pe[4:0]) && (cfg_index_i < PROGRAM_DEPTH)),
          .cfg_addr_i(cfg_index_i[4:0]),
          .cfg_word_i(cfg_word_i),
          .fetch_addr_i(pc_q[pe][4:0]),
          .fetch_word_o(instruction_word[pe]),
          .configured_o(),
          .tag_configure_i(launch_i && (instruction_count_q[pe] != 0)),
          .tag_configure_id_i(pe[3:0]),
          .tag_trip_count_i({2'd0, instruction_count_q[pe]}),
          .tag_frontier_i(6'd0),
          .tag_ready_i(1'b1),
          .tag_issue_i(issue_any[pe]),
          .tag_issue_id_i(decoded_tag),
          .tag_complete_i(tag_complete_q[pe]),
          .tag_complete_id_i(pe[3:0]),
          .tag_query_id_i(pe[3:0]),
          .tag_active_vector_o(tag_active[pe]),
          .tag_ready_vector_o(tag_ready[pe]),
          .tag_done_vector_o(tag_done[pe]),
          .tag_query_trip_count_o(),
          .tag_query_frontier_o(),
          .pipeline_class_i(pipeline_class[pe]),
          .pipeline_ready_i(4'hf),
          .issue_valid_o(control_issue_valid[pe]),
          .issue_tag_o(control_issue_tag[pe]),
          .rf_read_addr_a_i(rf_read_a[pe]),
          .rf_read_addr_b_i(rf_read_b[pe]),
          .rf_read_addr_c_i(rf_read_c[pe]),
          .rf_read_data_a_o(rf_a[pe]),
          .rf_read_data_b_o(rf_b[pe]),
          .rf_read_data_c_o(rf_c[pe]),
          .rf_write_enable_i(rf_write_enable[pe]),
          .rf_write_addr_i(rf_write_addr[pe]),
          .rf_write_data_i(rf_write_data[pe]),
          .network_valid_i(network_valid[pe]),
          .network_dx_i(network_dx[pe]),
          .network_dy_i(network_dy[pe]),
          .network_destination_register_i(network_dst[pe]),
          .network_tag_i(network_tag[pe]),
          .network_payload_i(network_payload[pe]),
          .network_valid_o(network_valid_unused),
          .network_dx_o(network_dx_unused),
          .network_dy_o(network_dy_unused),
          .network_route_o(network_route_unused),
          .network_consumed_hops_o(network_hops_unused),
          .network_delivered_o(network_delivered_unused),
          .network_payload_o(network_payload_unused),
          .fu_valid_i(fu_valid[pe]),
          .fu_op_i(fu_op[pe]),
          .fu_vector_a_i(rf_a[pe]),
          .fu_vector_b_i(rf_b[pe]),
          .fu_vector_c_i(rf_c[pe]),
          .fu_valid_o(fu_result_valid[pe]),
          .fu_vector_result_o(fu_result[pe]),
          .fu_illegal_o(fu_illegal[pe])
      );
    end
  endgenerate

  assign spm_req_valid_o = spm_select_valid;
  assign spm_req_write_o = spm_select_write;
  assign spm_req_addr_o = spm_select_addr;
  assign spm_req_wdata_o = spm_select_wdata;
  assign busy_o = running_q;
  assign stat_cycles_o = stat_cycles_q;
  assign stat_instructions_o = stat_instructions_q;
  assign stat_load_o = stat_load_q;
  assign stat_store_o = stat_store_q;
  assign stat_compute_o = stat_compute_q;
  assign stat_xfer_o = stat_xfer_q;
  assign stat_stall_o = stat_stall_q;
  assign stat_hops_o = stat_hops_q;
  assign stat_conflicts_o = stat_conflicts_q;

  always @* begin
    spm_select_valid = 1'b0;
    spm_select_pe = 4'd0;
    spm_select_write = 1'b0;
    spm_select_addr = 8'd0;
    spm_select_wdata = {VECTOR_BITS{1'b0}};
    backend_has_work = spm_pending_q;
    cycle_instruction_issues = 64'd0;
    cycle_load_issues = 64'd0;
    cycle_store_issues = 64'd0;
    cycle_compute_issues = 64'd0;
    cycle_xfer_issues = 64'd0;
    cycle_stalls = 64'd0;
    cycle_hops = 64'd0;
    cycle_conflicts = 64'd0;

    for (comb_pe = 0; comb_pe < PE_COUNT; comb_pe = comb_pe + 1) begin
      rf_read_a[comb_pe] = instruction_word[comb_pe][49:46];
      rf_read_b[comb_pe] = instruction_word[comb_pe][45:42];
      rf_read_c[comb_pe] = instruction_word[comb_pe][41:38];
      rf_write_enable[comb_pe] = 1'b0;
      rf_write_addr[comb_pe] = 4'd0;
      rf_write_data[comb_pe] = {VECTOR_BITS{1'b0}};
      fu_valid[comb_pe] = 1'b0;
      fu_op[comb_pe] = instruction_word[comb_pe][63:60];
      pipeline_class[comb_pe] = 32'd0;
      pipeline_class[comb_pe][2*comb_pe +: 2] = instruction_word[comb_pe][55:54];
      network_valid[comb_pe] = 1'b0;
      network_dx[comb_pe] = 5'sd0;
      network_dy[comb_pe] = 5'sd0;
      network_dst[comb_pe] = 4'd0;
      network_tag[comb_pe] = 4'd0;
      network_payload[comb_pe] = 64'd0;
      issue_any[comb_pe] = 1'b0;
      issue_xfer[comb_pe] = 1'b0;
      operands_are_ready[comb_pe] = sources_ready(
          instruction_word[comb_pe][63:60],
          rf_valid_q[comb_pe],
          instruction_word[comb_pe][49:46],
          instruction_word[comb_pe][45:42],
          instruction_word[comb_pe][41:38]
      );
      compute_commit[comb_pe] = (state_q[comb_pe] == ST_COMPUTE)
          && (compute_countdown_q[comb_pe] == 1);
      packet_route_grant[comb_pe] = packet_active_q[comb_pe];
      packet_will_deliver[comb_pe] = 1'b0;
      packet_delivery_accept[comb_pe] = 1'b0;
      packet_next_current[comb_pe] = packet_current_q[comb_pe];
      packet_next_dx[comb_pe] = packet_dx_q[comb_pe];
      packet_next_dy[comb_pe] = packet_dy_q[comb_pe];
      packet_step[comb_pe] = 2'd0;
      if ((state_q[comb_pe] != ST_IDLE) && (state_q[comb_pe] != ST_DONE))
        backend_has_work = 1'b1;
    end

    // One packet per physical router per cycle; lower source coordinate wins.
    for (comb_pe = 0; comb_pe < PE_COUNT; comb_pe = comb_pe + 1) begin
      for (comb_other = 0; comb_other < comb_pe; comb_other = comb_other + 1) begin
        if (packet_active_q[comb_other]
            && (packet_current_q[comb_other] == packet_current_q[comb_pe]))
          packet_route_grant[comb_pe] = 1'b0;
      end
      if (packet_active_q[comb_pe])
        backend_has_work = 1'b1;
      if (packet_route_grant[comb_pe]) begin
        if (packet_dx_q[comb_pe] > 0) begin
          packet_step[comb_pe] = (packet_dx_q[comb_pe] >= 2) ? 2'd2 : 2'd1;
          packet_next_dx[comb_pe] = packet_dx_q[comb_pe]
              - $signed({3'd0, packet_step[comb_pe]});
          packet_next_current[comb_pe] = packet_current_q[comb_pe] + packet_step[comb_pe];
        end else if (packet_dx_q[comb_pe] < 0) begin
          packet_step[comb_pe] = ((-packet_dx_q[comb_pe]) >= 2) ? 2'd2 : 2'd1;
          packet_next_dx[comb_pe] = packet_dx_q[comb_pe]
              + $signed({3'd0, packet_step[comb_pe]});
          packet_next_current[comb_pe] = packet_current_q[comb_pe] - packet_step[comb_pe];
        end else if (packet_dy_q[comb_pe] > 0) begin
          packet_step[comb_pe] = (packet_dy_q[comb_pe] >= 2) ? 2'd2 : 2'd1;
          packet_next_dy[comb_pe] = packet_dy_q[comb_pe]
              - $signed({3'd0, packet_step[comb_pe]});
          packet_next_current[comb_pe] = packet_current_q[comb_pe]
              + ({3'd0, packet_step[comb_pe]} << 2);
        end else if (packet_dy_q[comb_pe] < 0) begin
          packet_step[comb_pe] = ((-packet_dy_q[comb_pe]) >= 2) ? 2'd2 : 2'd1;
          packet_next_dy[comb_pe] = packet_dy_q[comb_pe]
              + $signed({3'd0, packet_step[comb_pe]});
          packet_next_current[comb_pe] = packet_current_q[comb_pe]
              - ({3'd0, packet_step[comb_pe]} << 2);
        end
        packet_will_deliver[comb_pe] = (packet_next_dx[comb_pe] == 0)
            && (packet_next_dy[comb_pe] == 0);
      end
    end

    // Arithmetic instructions issue independently in all physical PEs.
    for (comb_pe = 0; comb_pe < PE_COUNT; comb_pe = comb_pe + 1) begin
      if ((state_q[comb_pe] == ST_READY) && operands_are_ready[comb_pe]
          && (instruction_word[comb_pe][63:60] >= OP_FMA)
          && (instruction_word[comb_pe][63:60] <= OP_MUL)
          && (instruction_word[comb_pe][63:60] != OP_XFER)
          && control_issue_valid[comb_pe][instruction_word[comb_pe][55:54]]) begin
        fu_valid[comb_pe] = 1'b1;
        issue_any[comb_pe] = 1'b1;
      end
      if ((state_q[comb_pe] == ST_READY) && operands_are_ready[comb_pe]
          && (instruction_word[comb_pe][63:60] == OP_XFER)
          && !packet_active_q[comb_pe]
          && control_issue_valid[comb_pe][3]) begin
        issue_xfer[comb_pe] = 1'b1;
        issue_any[comb_pe] = 1'b1;
      end
    end

    // A single SPM vector port models bank/port contention.
    if (!spm_pending_q) begin
      for (comb_pe = 0; comb_pe < PE_COUNT; comb_pe = comb_pe + 1) begin
        if (!spm_select_valid && (state_q[comb_pe] == ST_READY)
            && operands_are_ready[comb_pe]
            && ((instruction_word[comb_pe][63:60] == OP_LOAD)
                || (instruction_word[comb_pe][63:60] == OP_STORE))
            && ((instruction_word[comb_pe][63:60] != OP_LOAD)
                || spm_valid_q[instruction_word[comb_pe][27:20]])
            && control_issue_valid[comb_pe][instruction_word[comb_pe][55:54]]) begin
          spm_select_valid = 1'b1;
          spm_select_pe = comb_pe[3:0];
          spm_select_write = instruction_word[comb_pe][63:60] == OP_STORE;
          spm_select_addr = instruction_word[comb_pe][27:20];
          spm_select_wdata = rf_a[comb_pe];
        end
      end
    end
    if (spm_select_valid && spm_req_ready_i)
      issue_any[spm_select_pe] = 1'b1;

    // Compute and DMA responses reserve destination write ports first.
    for (comb_pe = 0; comb_pe < PE_COUNT; comb_pe = comb_pe + 1) begin
      if (compute_commit[comb_pe]) begin
        rf_write_enable[comb_pe] = 1'b1;
        rf_write_addr[comb_pe] = pending_dst_q[comb_pe];
        rf_write_data[comb_pe] = fu_result[comb_pe];
      end
    end
    if (spm_pending_q && spm_rsp_valid_i) begin
      rf_write_enable[spm_pending_pe_q] = 1'b1;
      rf_write_addr[spm_pending_pe_q] = spm_pending_dst_q;
      rf_write_data[spm_pending_pe_q] = spm_rsp_rdata_i;
    end

    // Deliver routed packets only when the destination RF port is free.
    for (comb_pe = 0; comb_pe < PE_COUNT; comb_pe = comb_pe + 1) begin
      if (packet_route_grant[comb_pe] && packet_will_deliver[comb_pe]
          && !rf_write_enable[packet_next_current[comb_pe]]) begin
        rf_write_enable[packet_next_current[comb_pe]] = 1'b1;
        rf_write_addr[packet_next_current[comb_pe]] = packet_dst_q[comb_pe];
        rf_write_data[packet_next_current[comb_pe]] = packet_data_q[comb_pe];
        packet_delivery_accept[comb_pe] = 1'b1;
      end
    end

    // Exercise the router physically located at each current packet coordinate.
    for (comb_router = 0; comb_router < PE_COUNT; comb_router = comb_router + 1) begin
      for (comb_pe = 0; comb_pe < PE_COUNT; comb_pe = comb_pe + 1) begin
        if (packet_route_grant[comb_pe]
            && (packet_current_q[comb_pe] == comb_router[4:0])) begin
          network_valid[comb_router] = 1'b1;
          network_dx[comb_router] = packet_dx_q[comb_pe];
          network_dy[comb_router] = packet_dy_q[comb_pe];
          network_dst[comb_router] = packet_dst_q[comb_pe];
          network_tag[comb_router] = packet_tag_q[comb_pe];
          network_payload[comb_router] = packet_data_q[comb_pe][63:0];
        end
      end
    end

    for (comb_pe = 0; comb_pe < PE_COUNT; comb_pe = comb_pe + 1) begin
      if (fu_valid[comb_pe])
        cycle_compute_issues = cycle_compute_issues + 1'b1;
      if (issue_xfer[comb_pe])
        cycle_xfer_issues = cycle_xfer_issues + 1'b1;
      if ((state_q[comb_pe] == ST_READY) && !issue_any[comb_pe])
        cycle_stalls = cycle_stalls + 1'b1;
      if (packet_route_grant[comb_pe]) begin
        if (packet_will_deliver[comb_pe]) begin
          if (packet_delivery_accept[comb_pe])
            cycle_hops = cycle_hops + packet_step[comb_pe];
          else
            cycle_conflicts = cycle_conflicts + 1'b1;
        end else begin
          cycle_hops = cycle_hops + packet_step[comb_pe];
        end
      end else if (packet_active_q[comb_pe]) begin
        cycle_conflicts = cycle_conflicts + 1'b1;
      end
    end
    if (spm_select_valid && spm_req_ready_i) begin
      if (spm_select_write)
        cycle_store_issues = 64'd1;
      else
        cycle_load_issues = 64'd1;
    end
    cycle_instruction_issues = cycle_compute_issues + cycle_xfer_issues
        + cycle_load_issues + cycle_store_issues;
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      running_q <= 1'b0;
      done_o <= 1'b0;
      spm_pending_q <= 1'b0;
      spm_pending_pe_q <= 4'd0;
      spm_pending_dst_q <= 4'd0;
      spm_valid_q <= {SPM_VECTORS{1'b0}};
      stat_cycles_q <= 64'd0;
      stat_instructions_q <= 64'd0;
      stat_load_q <= 64'd0;
      stat_store_q <= 64'd0;
      stat_compute_q <= 64'd0;
      stat_xfer_q <= 64'd0;
      stat_stall_q <= 64'd0;
      stat_hops_q <= 64'd0;
      stat_conflicts_q <= 64'd0;
      for (reset_index = 0; reset_index < PE_COUNT; reset_index = reset_index + 1) begin
        instruction_count_q[reset_index] <= 6'd0;
        pc_q[reset_index] <= 6'd0;
        state_q[reset_index] <= ST_IDLE;
        compute_countdown_q[reset_index] <= 6'd0;
        pending_dst_q[reset_index] <= 4'd0;
        rf_valid_q[reset_index] <= 16'd0;
        packet_active_q[reset_index] <= 1'b0;
        packet_current_q[reset_index] <= 5'd0;
        packet_dx_q[reset_index] <= 5'sd0;
        packet_dy_q[reset_index] <= 5'sd0;
        packet_dst_q[reset_index] <= 4'd0;
        packet_tag_q[reset_index] <= 4'd0;
        packet_data_q[reset_index] <= {VECTOR_BITS{1'b0}};
        tag_complete_q[reset_index] <= 1'b0;
      end
    end else begin
      done_o <= 1'b0;
      for (sequential_pe = 0; sequential_pe < PE_COUNT; sequential_pe = sequential_pe + 1)
        tag_complete_q[sequential_pe] <= 1'b0;

      if (cfg_valid_i && (cfg_pe_i == 5'd16) && (cfg_index_i < PE_COUNT))
        instruction_count_q[cfg_index_i[3:0]] <= cfg_word_i[5:0];

      if (launch_i) begin
        running_q <= 1'b1;
        spm_pending_q <= 1'b0;
        stat_cycles_q <= 64'd0;
        stat_instructions_q <= 64'd0;
        stat_load_q <= 64'd0;
        stat_store_q <= 64'd0;
        stat_compute_q <= 64'd0;
        stat_xfer_q <= 64'd0;
        stat_stall_q <= 64'd0;
        stat_hops_q <= 64'd0;
        stat_conflicts_q <= 64'd0;
        for (sequential_pe = 0; sequential_pe < SPM_VECTORS; sequential_pe = sequential_pe + 1)
          spm_valid_q[sequential_pe] <= sequential_pe < input_vectors_i;
        for (sequential_pe = 0; sequential_pe < PE_COUNT; sequential_pe = sequential_pe + 1) begin
          pc_q[sequential_pe] <= 6'd0;
          state_q[sequential_pe] <= (instruction_count_q[sequential_pe] == 0)
              ? ST_DONE : ST_READY;
          compute_countdown_q[sequential_pe] <= 6'd0;
          pending_dst_q[sequential_pe] <= 4'd0;
          rf_valid_q[sequential_pe] <= 16'd0;
          packet_active_q[sequential_pe] <= 1'b0;
          packet_current_q[sequential_pe] <= sequential_pe[4:0];
          packet_dx_q[sequential_pe] <= 5'sd0;
          packet_dy_q[sequential_pe] <= 5'sd0;
        end
      end else begin
        if (running_q)
          stat_cycles_q <= stat_cycles_q + 1'b1;
        if (running_q) begin
          stat_instructions_q <= stat_instructions_q + cycle_instruction_issues;
          stat_load_q <= stat_load_q + cycle_load_issues;
          stat_store_q <= stat_store_q + cycle_store_issues;
          stat_compute_q <= stat_compute_q + cycle_compute_issues;
          stat_xfer_q <= stat_xfer_q + cycle_xfer_issues;
          stat_stall_q <= stat_stall_q + cycle_stalls;
          stat_hops_q <= stat_hops_q + cycle_hops;
          stat_conflicts_q <= stat_conflicts_q + cycle_conflicts;
        end
        if (running_q && !backend_has_work) begin
          running_q <= 1'b0;
          done_o <= 1'b1;
`ifndef SYNTHESIS
          $display("MLX_BACKEND_DONE backend=rtl cycle=%0d instructions=%0d stalls=%0d hops=%0d conflicts=%0d",
                   stat_cycles_q, stat_instructions_q, stat_stall_q,
                   stat_hops_q, stat_conflicts_q);
`endif
        end

        // Instruction issue and local FU completion.
        for (sequential_pe = 0; sequential_pe < PE_COUNT; sequential_pe = sequential_pe + 1) begin
          if (fu_valid[sequential_pe]) begin
            state_q[sequential_pe] <= ST_COMPUTE;
            compute_countdown_q[sequential_pe]
                <= operation_latency(instruction_word[sequential_pe][63:60]);
            pending_dst_q[sequential_pe] <= instruction_word[sequential_pe][53:50];
            rf_valid_q[sequential_pe][instruction_word[sequential_pe][53:50]] <= 1'b0;
`ifndef SYNTHESIS
            $display("MLX_TRACE backend=rtl event=issue cycle=%0d pe=%0d pc=%0d op=%0d tag=%0d",
                     stat_cycles_q, sequential_pe, pc_q[sequential_pe],
                     instruction_word[sequential_pe][63:60],
                     instruction_word[sequential_pe][59:56]);
`endif
          end else if (state_q[sequential_pe] == ST_COMPUTE) begin
            if (compute_countdown_q[sequential_pe] > 1) begin
              compute_countdown_q[sequential_pe]
                  <= compute_countdown_q[sequential_pe] - 1'b1;
            end else begin
              rf_valid_q[sequential_pe][pending_dst_q[sequential_pe]] <= 1'b1;
`ifndef SYNTHESIS
              $display("MLX_TRACE backend=rtl event=complete cycle=%0d pe=%0d pc=%0d op=%0d",
                       stat_cycles_q, sequential_pe, pc_q[sequential_pe],
                       instruction_word[sequential_pe][63:60]);
`endif
              if (pc_q[sequential_pe] + 1 >= instruction_count_q[sequential_pe]) begin
                state_q[sequential_pe] <= ST_DONE;
                tag_complete_q[sequential_pe] <= 1'b1;
              end else begin
                pc_q[sequential_pe] <= pc_q[sequential_pe] + 1'b1;
                state_q[sequential_pe] <= ST_READY;
              end
            end
          end

          if (issue_xfer[sequential_pe]) begin
            packet_active_q[sequential_pe] <= 1'b1;
            packet_current_q[sequential_pe] <= sequential_pe[4:0];
            packet_dx_q[sequential_pe] <= instruction_word[sequential_pe][37:33];
            packet_dy_q[sequential_pe] <= instruction_word[sequential_pe][32:28];
            packet_dst_q[sequential_pe] <= instruction_word[sequential_pe][53:50];
            packet_tag_q[sequential_pe] <= instruction_word[sequential_pe][59:56];
            packet_data_q[sequential_pe] <= rf_a[sequential_pe];
            state_q[sequential_pe] <= ST_XFER;
`ifndef SYNTHESIS
            $display("MLX_TRACE backend=rtl event=issue cycle=%0d pe=%0d pc=%0d op=8 tag=%0d",
                     stat_cycles_q, sequential_pe, pc_q[sequential_pe],
                     instruction_word[sequential_pe][59:56]);
`endif
          end

        end

        // Shared SPM request/response path.
        if (spm_select_valid && spm_req_ready_i) begin
          if (spm_select_write) begin
            spm_valid_q[spm_select_addr] <= 1'b1;
`ifndef SYNTHESIS
            $display("MLX_TRACE backend=rtl event=issue cycle=%0d pe=%0d pc=%0d op=1 tag=%0d",
                     stat_cycles_q, spm_select_pe, pc_q[spm_select_pe],
                     instruction_word[spm_select_pe][59:56]);
            $display("MLX_TRACE backend=rtl event=complete cycle=%0d pe=%0d pc=%0d op=1",
                     stat_cycles_q + 1'b1, spm_select_pe, pc_q[spm_select_pe]);
`endif
            if (pc_q[spm_select_pe] + 1 >= instruction_count_q[spm_select_pe]) begin
              state_q[spm_select_pe] <= ST_DONE;
              tag_complete_q[spm_select_pe] <= 1'b1;
            end else begin
              pc_q[spm_select_pe] <= pc_q[spm_select_pe] + 1'b1;
              state_q[spm_select_pe] <= ST_READY;
            end
          end else begin
            spm_pending_q <= 1'b1;
            spm_pending_pe_q <= spm_select_pe;
            spm_pending_dst_q <= instruction_word[spm_select_pe][53:50];
            rf_valid_q[spm_select_pe][instruction_word[spm_select_pe][53:50]] <= 1'b0;
            state_q[spm_select_pe] <= ST_LOAD;
`ifndef SYNTHESIS
            $display("MLX_TRACE backend=rtl event=issue cycle=%0d pe=%0d pc=%0d op=0 tag=%0d",
                     stat_cycles_q, spm_select_pe, pc_q[spm_select_pe],
                     instruction_word[spm_select_pe][59:56]);
`endif
          end
        end
        if (spm_pending_q && spm_rsp_valid_i) begin
          spm_pending_q <= 1'b0;
          rf_valid_q[spm_pending_pe_q][spm_pending_dst_q] <= 1'b1;
`ifndef SYNTHESIS
          $display("MLX_TRACE backend=rtl event=complete cycle=%0d pe=%0d pc=%0d op=0",
                   stat_cycles_q, spm_pending_pe_q, pc_q[spm_pending_pe_q]);
`endif
          if (pc_q[spm_pending_pe_q] + 1 >= instruction_count_q[spm_pending_pe_q]) begin
            state_q[spm_pending_pe_q] <= ST_DONE;
            tag_complete_q[spm_pending_pe_q] <= 1'b1;
          end else begin
            pc_q[spm_pending_pe_q] <= pc_q[spm_pending_pe_q] + 1'b1;
            state_q[spm_pending_pe_q] <= ST_READY;
          end
        end

        // Mesh routing and destination flow control.
        for (sequential_pe = 0; sequential_pe < PE_COUNT; sequential_pe = sequential_pe + 1) begin
          if (packet_route_grant[sequential_pe]) begin
            if (packet_will_deliver[sequential_pe]) begin
              if (packet_delivery_accept[sequential_pe]) begin
                packet_active_q[sequential_pe] <= 1'b0;
                rf_valid_q[packet_next_current[sequential_pe]][packet_dst_q[sequential_pe]]
                    <= 1'b1;
`ifndef SYNTHESIS
                $display("MLX_TRACE backend=rtl event=complete cycle=%0d pe=%0d pc=%0d op=8 dst_pe=%0d hops=%0d",
                         stat_cycles_q, sequential_pe, pc_q[sequential_pe],
                         packet_next_current[sequential_pe], packet_step[sequential_pe]);
`endif
                if (pc_q[sequential_pe] + 1 >= instruction_count_q[sequential_pe]) begin
                  state_q[sequential_pe] <= ST_DONE;
                  tag_complete_q[sequential_pe] <= 1'b1;
                end else begin
                  pc_q[sequential_pe] <= pc_q[sequential_pe] + 1'b1;
                  state_q[sequential_pe] <= ST_READY;
                end
              end
            end else begin
              packet_current_q[sequential_pe] <= packet_next_current[sequential_pe];
              packet_dx_q[sequential_pe] <= packet_next_dx[sequential_pe];
              packet_dy_q[sequential_pe] <= packet_next_dy[sequential_pe];
            end
          end
        end
      end
    end
  end
endmodule

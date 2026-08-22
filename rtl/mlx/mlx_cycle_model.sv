`timescale 1ns/1ps

// Fast architectural cycle model.  It interprets the same per-PE spatial
// program as the RTL array, but uses one shared SIMD functional service and a
// ready-tag scan instead of physically instantiating sixteen PEs and routers.
// This is intentionally a distinct timing model: instructions are serialized,
// xfers use abstract skip-hop delay, and SPM still uses the shared request port.
module mlx_cycle_model #(
    parameter SIMD_WIDTH = 32,
    parameter DATA_BITS = 16,
    parameter VECTOR_BITS = SIMD_WIDTH * DATA_BITS,
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
    output reg                    spm_req_valid_o,
    input  wire                   spm_req_ready_i,
    output reg                    spm_req_write_o,
    output reg [7:0]              spm_req_addr_o,
    output reg [VECTOR_BITS-1:0]  spm_req_wdata_o,
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
  localparam ST_RUN = 3'd1;
  localparam ST_COMPUTE = 3'd2;
  localparam ST_LOAD = 3'd3;
  localparam ST_XFER = 3'd4;

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

  reg [63:0] program_q [0:PE_COUNT*PROGRAM_DEPTH-1];
  reg [5:0] instruction_count_q [0:PE_COUNT-1];
  reg [5:0] pc_q [0:PE_COUNT-1];
  wire [63:0] current_word [0:PE_COUNT-1];
  reg [VECTOR_BITS-1:0] rf_q [0:PE_COUNT-1][0:15];
  reg [15:0] rf_valid_q [0:PE_COUNT-1];
  reg [SPM_VECTORS-1:0] spm_valid_q;
  reg [2:0] state_q;
  reg running_q;

  reg [3:0] selected_pe;
  reg selected_valid;
  reg selected_ready;
  reg [63:0] selected_word;
  reg [63:0] blocked_tags;
  reg [63:0] ready_tags;
  reg all_complete;

  reg [3:0] pending_pe_q;
  reg [3:0] pending_dst_q;
  reg [3:0] pending_target_pe_q;
  reg [5:0] pending_countdown_q;
  reg [VECTOR_BITS-1:0] pending_xfer_data_q;

  reg fu_valid;
  wire fu_valid_out;
  wire [VECTOR_BITS-1:0] fu_result;
  wire fu_illegal;
  wire [255:0] fu_high_precision_unused;

  reg [63:0] stat_cycles_q;
  reg [63:0] stat_instructions_q;
  reg [63:0] stat_load_q;
  reg [63:0] stat_store_q;
  reg [63:0] stat_compute_q;
  reg [63:0] stat_xfer_q;
  reg [63:0] stat_stall_q;
  reg [63:0] stat_hops_q;
  reg [63:0] stat_conflicts_q;

  integer scan_pe;
  integer launch_index;

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

  function [5:0] abstract_hops;
    input signed [4:0] dx;
    input signed [4:0] dy;
    reg [4:0] abs_dx;
    reg [4:0] abs_dy;
    begin
      abs_dx = dx < 0 ? -dx : dx;
      abs_dy = dy < 0 ? -dy : dy;
      abstract_hops = ((abs_dx + 1'b1) >> 1) + ((abs_dy + 1'b1) >> 1);
      if (abstract_hops == 0)
        abstract_hops = 1;
    end
  endfunction

  function [3:0] routed_target;
    input [3:0] source;
    input signed [4:0] dx;
    input signed [4:0] dy;
    integer target;
    begin
      target = source + dx + 4 * dy;
      routed_target = target[3:0];
    end
  endfunction

  genvar pe;
  generate
    for (pe = 0; pe < PE_COUNT; pe = pe + 1) begin : GENERATE_CURRENT_WORDS
      assign current_word[pe] = program_q[pe*PROGRAM_DEPTH + pc_q[pe][4:0]];
    end
  endgenerate

  mlx_fu #(
      .SIMD_WIDTH(SIMD_WIDTH),
      .FULL_FEATURES(1),
      .TRANS_LANES(SIMD_WIDTH / 4)
  ) shared_functional_service (
      .clk(clk),
      .rst_n(rst_n),
      .valid_i(fu_valid),
      .op_i(selected_word[63:60]),
      .vector_a_i(rf_q[selected_pe][selected_word[49:46]]),
      .vector_b_i(rf_q[selected_pe][selected_word[45:42]]),
      .vector_c_i(rf_q[selected_pe][selected_word[41:38]]),
      .valid_o(fu_valid_out),
      .vector_result_o(fu_result),
      .illegal_o(fu_illegal),
      .high_precision_result_o(fu_high_precision_unused)
  );

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
    selected_pe = 4'd0;
    selected_valid = 1'b0;
    selected_ready = 1'b0;
    selected_word = 64'd0;
    blocked_tags = 64'd0;
    ready_tags = 64'd0;
    all_complete = 1'b1;
    for (scan_pe = 0; scan_pe < PE_COUNT; scan_pe = scan_pe + 1) begin
      if (pc_q[scan_pe] < instruction_count_q[scan_pe]) begin
        all_complete = 1'b0;
        if (sources_ready(
            current_word[scan_pe][63:60],
            rf_valid_q[scan_pe],
            current_word[scan_pe][49:46],
            current_word[scan_pe][45:42],
            current_word[scan_pe][41:38])
            && ((current_word[scan_pe][63:60] != OP_LOAD)
                || spm_valid_q[current_word[scan_pe][26:20]])) begin
          ready_tags = ready_tags + 1'b1;
          if (!selected_valid) begin
            selected_valid = 1'b1;
            selected_ready = 1'b1;
            selected_pe = scan_pe[3:0];
            selected_word = current_word[scan_pe];
          end
        end else begin
          blocked_tags = blocked_tags + 1'b1;
        end
      end
    end

    fu_valid = (state_q == ST_RUN) && selected_valid
        && (selected_word[63:60] >= OP_FMA)
        && (selected_word[63:60] <= OP_MUL)
        && (selected_word[63:60] != OP_XFER);
    spm_req_valid_o = (state_q == ST_RUN) && selected_valid
        && ((selected_word[63:60] == OP_LOAD)
            || (selected_word[63:60] == OP_STORE));
    spm_req_write_o = selected_word[63:60] == OP_STORE;
    spm_req_addr_o = selected_word[27:20];
    spm_req_wdata_o = rf_q[selected_pe][selected_word[49:46]];
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state_q <= ST_IDLE;
      running_q <= 1'b0;
      done_o <= 1'b0;
      spm_valid_q <= {SPM_VECTORS{1'b0}};
      pending_pe_q <= 4'd0;
      pending_dst_q <= 4'd0;
      pending_target_pe_q <= 4'd0;
      pending_countdown_q <= 6'd0;
      pending_xfer_data_q <= {VECTOR_BITS{1'b0}};
      stat_cycles_q <= 64'd0;
      stat_instructions_q <= 64'd0;
      stat_load_q <= 64'd0;
      stat_store_q <= 64'd0;
      stat_compute_q <= 64'd0;
      stat_xfer_q <= 64'd0;
      stat_stall_q <= 64'd0;
      stat_hops_q <= 64'd0;
      stat_conflicts_q <= 64'd0;
      for (launch_index = 0; launch_index < PE_COUNT; launch_index = launch_index + 1) begin
        instruction_count_q[launch_index] <= 6'd0;
        pc_q[launch_index] <= 6'd0;
        rf_valid_q[launch_index] <= 16'd0;
      end
    end else begin
      done_o <= 1'b0;
      if (cfg_valid_i && (cfg_pe_i < PE_COUNT) && (cfg_index_i < PROGRAM_DEPTH))
        program_q[cfg_pe_i*PROGRAM_DEPTH + cfg_index_i] <= cfg_word_i;
      if (cfg_valid_i && (cfg_pe_i == 5'd16) && (cfg_index_i < PE_COUNT))
        instruction_count_q[cfg_index_i[3:0]] <= cfg_word_i[5:0];

      if (launch_i) begin
        state_q <= ST_RUN;
        running_q <= 1'b1;
        stat_cycles_q <= 64'd0;
        stat_instructions_q <= 64'd0;
        stat_load_q <= 64'd0;
        stat_store_q <= 64'd0;
        stat_compute_q <= 64'd0;
        stat_xfer_q <= 64'd0;
        stat_stall_q <= 64'd0;
        stat_hops_q <= 64'd0;
        stat_conflicts_q <= 64'd0;
        for (launch_index = 0; launch_index < SPM_VECTORS; launch_index = launch_index + 1)
          spm_valid_q[launch_index] <= launch_index < input_vectors_i;
        for (launch_index = 0; launch_index < PE_COUNT; launch_index = launch_index + 1) begin
          pc_q[launch_index] <= 6'd0;
          rf_valid_q[launch_index] <= 16'd0;
        end
      end else if (running_q) begin
        stat_cycles_q <= stat_cycles_q + 1'b1;
        case (state_q)
          ST_RUN: begin
            stat_stall_q <= stat_stall_q + blocked_tags;
            if (ready_tags > 1)
              stat_conflicts_q <= stat_conflicts_q + ready_tags - 1'b1;
            if (all_complete) begin
              running_q <= 1'b0;
              done_o <= 1'b1;
              state_q <= ST_IDLE;
`ifndef SYNTHESIS
              $display("MLX_BACKEND_DONE backend=cycle cycle=%0d instructions=%0d stalls=%0d hops=%0d conflicts=%0d",
                       stat_cycles_q, stat_instructions_q, stat_stall_q,
                       stat_hops_q, stat_conflicts_q);
`endif
            end else if (selected_valid) begin
              if (fu_valid) begin
                pending_pe_q <= selected_pe;
                pending_dst_q <= selected_word[53:50];
                pending_countdown_q <= operation_latency(selected_word[63:60]);
                rf_valid_q[selected_pe][selected_word[53:50]] <= 1'b0;
                stat_instructions_q <= stat_instructions_q + 1'b1;
                stat_compute_q <= stat_compute_q + 1'b1;
                state_q <= ST_COMPUTE;
              end else if ((selected_word[63:60] == OP_LOAD)
                           && spm_req_valid_o && spm_req_ready_i) begin
                pending_pe_q <= selected_pe;
                pending_dst_q <= selected_word[53:50];
                rf_valid_q[selected_pe][selected_word[53:50]] <= 1'b0;
                stat_instructions_q <= stat_instructions_q + 1'b1;
                stat_load_q <= stat_load_q + 1'b1;
                state_q <= ST_LOAD;
              end else if ((selected_word[63:60] == OP_STORE)
                           && spm_req_valid_o && spm_req_ready_i) begin
                spm_valid_q[selected_word[26:20]] <= 1'b1;
                stat_instructions_q <= stat_instructions_q + 1'b1;
                stat_store_q <= stat_store_q + 1'b1;
                pc_q[selected_pe] <= pc_q[selected_pe] + 1'b1;
              end else if (selected_word[63:60] == OP_XFER) begin
                pending_pe_q <= selected_pe;
                pending_dst_q <= selected_word[53:50];
                pending_target_pe_q <= routed_target(
                    selected_pe, selected_word[37:33], selected_word[32:28]);
                pending_countdown_q <= abstract_hops(
                    selected_word[37:33], selected_word[32:28]);
                pending_xfer_data_q <= rf_q[selected_pe][selected_word[49:46]];
                stat_instructions_q <= stat_instructions_q + 1'b1;
                stat_xfer_q <= stat_xfer_q + 1'b1;
                stat_hops_q <= stat_hops_q + {
                    58'd0, abstract_hops(selected_word[37:33], selected_word[32:28])
                };
                state_q <= ST_XFER;
              end
`ifndef SYNTHESIS
              if (fu_valid || (spm_req_valid_o && spm_req_ready_i)
                  || (selected_word[63:60] == OP_XFER))
                $display("MLX_TRACE backend=cycle event=issue cycle=%0d pe=%0d pc=%0d op=%0d tag=%0d",
                         stat_cycles_q, selected_pe, pc_q[selected_pe],
                         selected_word[63:60], selected_word[59:56]);
`endif
            end
          end
          ST_COMPUTE: begin
            if (pending_countdown_q > 1) begin
              pending_countdown_q <= pending_countdown_q - 1'b1;
            end else begin
              rf_q[pending_pe_q][pending_dst_q] <= fu_result;
              rf_valid_q[pending_pe_q][pending_dst_q] <= 1'b1;
              pc_q[pending_pe_q] <= pc_q[pending_pe_q] + 1'b1;
              state_q <= ST_RUN;
`ifndef SYNTHESIS
              $display("MLX_TRACE backend=cycle event=complete cycle=%0d pe=%0d op=compute",
                       stat_cycles_q, pending_pe_q);
`endif
            end
          end
          ST_LOAD: begin
            if (spm_rsp_valid_i) begin
              rf_q[pending_pe_q][pending_dst_q] <= spm_rsp_rdata_i;
              rf_valid_q[pending_pe_q][pending_dst_q] <= 1'b1;
              pc_q[pending_pe_q] <= pc_q[pending_pe_q] + 1'b1;
              state_q <= ST_RUN;
            end
          end
          ST_XFER: begin
            if (pending_countdown_q > 1) begin
              pending_countdown_q <= pending_countdown_q - 1'b1;
            end else begin
              rf_q[pending_target_pe_q][pending_dst_q] <= pending_xfer_data_q;
              rf_valid_q[pending_target_pe_q][pending_dst_q] <= 1'b1;
              pc_q[pending_pe_q] <= pc_q[pending_pe_q] + 1'b1;
              state_q <= ST_RUN;
`ifndef SYNTHESIS
              $display("MLX_TRACE backend=cycle event=complete cycle=%0d pe=%0d op=xfer dst_pe=%0d",
                       stat_cycles_q, pending_pe_q, pending_target_pe_q);
`endif
            end
          end
          default: state_q <= ST_RUN;
        endcase
      end
    end
  end
endmodule

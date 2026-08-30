`timescale 1ns/1ps

// Experimental paper-aligned 4x4 hierarchy.  Each autonomous tile owns its
// local RF/FU/writeback paths and one registered packet buffer.  The top only
// arbitrates the shared SPM port and wires fixed distance-1/distance-2 mesh
// links.  It intentionally coexists with mlx_array_4x4 until workload and PPA
// comparisons establish that it can replace the centralized implementation.
module mlx_array_4x4_distributed #(
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
  wire tile_spm_valid [0:PE_COUNT-1];
  wire tile_spm_write [0:PE_COUNT-1];
  wire [7:0] tile_spm_addr [0:PE_COUNT-1];
  wire [VECTOR_BITS-1:0] tile_spm_wdata [0:PE_COUNT-1];
  reg tile_spm_grant [0:PE_COUNT-1];
  reg tile_spm_rsp_valid [0:PE_COUNT-1];

  reg route_in_valid [0:PE_COUNT-1];
  reg signed [4:0] route_in_dx [0:PE_COUNT-1];
  reg signed [4:0] route_in_dy [0:PE_COUNT-1];
  reg [3:0] route_in_destination_register [0:PE_COUNT-1];
  reg [3:0] route_in_tag [0:PE_COUNT-1];
  reg [3:0] route_in_source [0:PE_COUNT-1];
  reg [VECTOR_BITS-1:0] route_in_data [0:PE_COUNT-1];
  wire route_in_ready [0:PE_COUNT-1];

  wire route_out_valid [0:PE_COUNT-1];
  wire [4:0] route_out_target [0:PE_COUNT-1];
  wire signed [4:0] route_out_dx [0:PE_COUNT-1];
  wire signed [4:0] route_out_dy [0:PE_COUNT-1];
  wire [3:0] route_out_destination_register [0:PE_COUNT-1];
  wire [3:0] route_out_tag [0:PE_COUNT-1];
  wire [3:0] route_out_source [0:PE_COUNT-1];
  wire [VECTOR_BITS-1:0] route_out_data [0:PE_COUNT-1];
  wire [1:0] route_out_hops [0:PE_COUNT-1];
  reg route_out_grant [0:PE_COUNT-1];

  wire xfer_complete_valid [0:PE_COUNT-1];
  wire [3:0] xfer_complete_source [0:PE_COUNT-1];
  reg xfer_complete [0:PE_COUNT-1];

  wire tile_execution_active [0:PE_COUNT-1];
  wire tile_router_active [0:PE_COUNT-1];
  wire tile_instruction_issue [0:PE_COUNT-1];
  wire tile_load_issue [0:PE_COUNT-1];
  wire tile_store_issue [0:PE_COUNT-1];
  wire tile_compute_issue [0:PE_COUNT-1];
  wire tile_xfer_issue [0:PE_COUNT-1];
  wire tile_stall [0:PE_COUNT-1];
  wire tile_delivery_conflict [0:PE_COUNT-1];

  wire [PE_COUNT-1:0] route_candidate [0:PE_COUNT-1];

  reg spm_select_valid;
  reg [3:0] spm_select_pe;
  reg spm_select_write;
  reg [7:0] spm_select_addr;
  reg [VECTOR_BITS-1:0] spm_select_wdata;
  reg spm_pending_q;
  reg [3:0] spm_pending_pe_q;
  reg [SPM_VECTORS-1:0] spm_valid_q;
  reg running_q;

  reg [63:0] stat_cycles_q;
  reg [63:0] stat_instructions_q;
  reg [63:0] stat_load_q;
  reg [63:0] stat_store_q;
  reg [63:0] stat_compute_q;
  reg [63:0] stat_xfer_q;
  reg [63:0] stat_stall_q;
  reg [63:0] stat_hops_q;
  reg [63:0] stat_conflicts_q;

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
  integer comb_source;
  integer comb_destination;
  integer reset_index;

  genvar tile;
  generate
    for (tile = 0; tile < PE_COUNT; tile = tile + 1) begin : GENERATE_TILES
`ifdef MLX_PPA_TILE_MACRO
      mlx_array_pe_tile physical_tile (
`else
      mlx_array_pe_tile #(
          .SIMD_WIDTH(SIMD_WIDTH),
          .DATA_BITS(DATA_BITS),
          .VECTOR_BITS(VECTOR_BITS),
          .TRANS_LANES(TRANS_LANES),
          .PROGRAM_DEPTH(PROGRAM_DEPTH)
      ) physical_tile (
`endif
          .clk(clk),
          .rst_n(rst_n),
          .cfg_valid_i(cfg_valid_i && (cfg_pe_i == tile[4:0])
              && (cfg_index_i < PROGRAM_DEPTH)),
          .cfg_addr_i(cfg_index_i[4:0]),
          .cfg_word_i(cfg_word_i),
          .cfg_instruction_count_valid_i(cfg_valid_i && (cfg_pe_i == 5'd16)
              && (cfg_index_i == tile[5:0])),
          .cfg_instruction_count_i(cfg_word_i[5:0]),
          .launch_i(launch_i),
          .tile_id_i(tile[3:0]),
          .spm_req_valid_o(tile_spm_valid[tile]),
          .spm_req_write_o(tile_spm_write[tile]),
          .spm_req_addr_o(tile_spm_addr[tile]),
          .spm_req_wdata_o(tile_spm_wdata[tile]),
          .spm_req_grant_i(tile_spm_grant[tile]),
          .spm_rsp_valid_i(tile_spm_rsp_valid[tile]),
          .spm_rsp_rdata_i(spm_rsp_rdata_i),
          .route_in_valid_i(route_in_valid[tile]),
          .route_in_dx_i(route_in_dx[tile]),
          .route_in_dy_i(route_in_dy[tile]),
          .route_in_destination_register_i(route_in_destination_register[tile]),
          .route_in_tag_i(route_in_tag[tile]),
          .route_in_source_i(route_in_source[tile]),
          .route_in_data_i(route_in_data[tile]),
          .route_in_ready_o(route_in_ready[tile]),
          .route_out_valid_o(route_out_valid[tile]),
          .route_out_target_o(route_out_target[tile]),
          .route_out_dx_o(route_out_dx[tile]),
          .route_out_dy_o(route_out_dy[tile]),
          .route_out_destination_register_o(route_out_destination_register[tile]),
          .route_out_tag_o(route_out_tag[tile]),
          .route_out_source_o(route_out_source[tile]),
          .route_out_data_o(route_out_data[tile]),
          .route_out_hops_o(route_out_hops[tile]),
          .route_out_grant_i(route_out_grant[tile]),
          .xfer_complete_i(xfer_complete[tile]),
          .xfer_complete_valid_o(xfer_complete_valid[tile]),
          .xfer_complete_source_o(xfer_complete_source[tile]),
          .execution_active_o(tile_execution_active[tile]),
          .router_active_o(tile_router_active[tile]),
          .instruction_issue_o(tile_instruction_issue[tile]),
          .load_issue_o(tile_load_issue[tile]),
          .store_issue_o(tile_store_issue[tile]),
          .compute_issue_o(tile_compute_issue[tile]),
          .xfer_issue_o(tile_xfer_issue[tile]),
          .stall_o(tile_stall[tile]),
          .delivery_conflict_o(tile_delivery_conflict[tile])
      );
    end
  endgenerate

  genvar destination;
  genvar source;
  generate
    for (destination = 0; destination < PE_COUNT;
         destination = destination + 1) begin : GENERATE_DESTINATIONS
      for (source = 0; source < PE_COUNT; source = source + 1) begin : GENERATE_SOURCES
        localparam integer SOURCE_X = source % 4;
        localparam integer SOURCE_Y = source / 4;
        localparam integer DESTINATION_X = destination % 4;
        localparam integer DESTINATION_Y = destination / 4;
        localparam integer X_DISTANCE = (SOURCE_X > DESTINATION_X)
            ? SOURCE_X - DESTINATION_X : DESTINATION_X - SOURCE_X;
        localparam integer Y_DISTANCE = (SOURCE_Y > DESTINATION_Y)
            ? SOURCE_Y - DESTINATION_Y : DESTINATION_Y - SOURCE_Y;
        localparam integer FIXED_LINK =
            ((SOURCE_Y == DESTINATION_Y) && (X_DISTANCE >= 1) && (X_DISTANCE <= 2))
            || ((SOURCE_X == DESTINATION_X) && (Y_DISTANCE >= 1) && (Y_DISTANCE <= 2));
        if (FIXED_LINK != 0) begin : LINK_EXISTS
          assign route_candidate[destination][source] = route_out_valid[source]
              && (route_out_target[source] == destination);
        end else begin : NO_LINK
          assign route_candidate[destination][source] = 1'b0;
        end
      end
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
    for (comb_pe = 0; comb_pe < PE_COUNT; comb_pe = comb_pe + 1) begin
      tile_spm_grant[comb_pe] = 1'b0;
      tile_spm_rsp_valid[comb_pe] = spm_pending_q && spm_rsp_valid_i
          && (spm_pending_pe_q == comb_pe[3:0]);
      if (!spm_pending_q && !spm_select_valid && tile_spm_valid[comb_pe]
          && (tile_spm_write[comb_pe] || spm_valid_q[tile_spm_addr[comb_pe]])) begin
        spm_select_valid = 1'b1;
        spm_select_pe = comb_pe[3:0];
        spm_select_write = tile_spm_write[comb_pe];
        spm_select_addr = tile_spm_addr[comb_pe];
        spm_select_wdata = tile_spm_wdata[comb_pe];
      end
    end
    if (spm_select_valid && spm_req_ready_i)
      tile_spm_grant[spm_select_pe] = 1'b1;
  end

  always @* begin
    for (comb_source = 0; comb_source < PE_COUNT; comb_source = comb_source + 1) begin
      route_out_grant[comb_source] = 1'b0;
      xfer_complete[comb_source] = 1'b0;
    end
    for (comb_destination = 0; comb_destination < PE_COUNT;
         comb_destination = comb_destination + 1) begin
      route_in_valid[comb_destination] = 1'b0;
      route_in_dx[comb_destination] = 5'sd0;
      route_in_dy[comb_destination] = 5'sd0;
      route_in_destination_register[comb_destination] = 4'd0;
      route_in_tag[comb_destination] = 4'd0;
      route_in_source[comb_destination] = 4'd0;
      route_in_data[comb_destination] = {VECTOR_BITS{1'b0}};
      for (comb_source = 0; comb_source < PE_COUNT; comb_source = comb_source + 1) begin
        if (!route_in_valid[comb_destination]
            && route_candidate[comb_destination][comb_source]
            && route_in_ready[comb_destination]) begin
          route_in_valid[comb_destination] = 1'b1;
          route_in_dx[comb_destination] = route_out_dx[comb_source];
          route_in_dy[comb_destination] = route_out_dy[comb_source];
          route_in_destination_register[comb_destination]
              = route_out_destination_register[comb_source];
          route_in_tag[comb_destination] = route_out_tag[comb_source];
          route_in_source[comb_destination] = route_out_source[comb_source];
          route_in_data[comb_destination] = route_out_data[comb_source];
          route_out_grant[comb_source] = 1'b1;
        end
      end
      if (xfer_complete_valid[comb_destination])
        xfer_complete[xfer_complete_source[comb_destination]] = 1'b1;
    end
  end

  always @* begin
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
      backend_has_work = backend_has_work || tile_execution_active[comb_pe]
          || tile_router_active[comb_pe];
      cycle_instruction_issues = cycle_instruction_issues
          + tile_instruction_issue[comb_pe];
      cycle_load_issues = cycle_load_issues + tile_load_issue[comb_pe];
      cycle_store_issues = cycle_store_issues + tile_store_issue[comb_pe];
      cycle_compute_issues = cycle_compute_issues + tile_compute_issue[comb_pe];
      cycle_xfer_issues = cycle_xfer_issues + tile_xfer_issue[comb_pe];
      cycle_stalls = cycle_stalls + tile_stall[comb_pe];
      if (route_out_grant[comb_pe])
        cycle_hops = cycle_hops + route_out_hops[comb_pe];
      if ((route_out_valid[comb_pe] && !route_out_grant[comb_pe])
          || tile_delivery_conflict[comb_pe])
        cycle_conflicts = cycle_conflicts + 1'b1;
    end
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      spm_pending_q <= 1'b0;
      spm_pending_pe_q <= 4'd0;
      spm_valid_q <= {SPM_VECTORS{1'b0}};
      running_q <= 1'b0;
      done_o <= 1'b0;
      stat_cycles_q <= 64'd0;
      stat_instructions_q <= 64'd0;
      stat_load_q <= 64'd0;
      stat_store_q <= 64'd0;
      stat_compute_q <= 64'd0;
      stat_xfer_q <= 64'd0;
      stat_stall_q <= 64'd0;
      stat_hops_q <= 64'd0;
      stat_conflicts_q <= 64'd0;
    end else begin
      done_o <= 1'b0;
      if (launch_i) begin
        spm_pending_q <= 1'b0;
        for (reset_index = 0; reset_index < SPM_VECTORS; reset_index = reset_index + 1)
          spm_valid_q[reset_index] <= reset_index < input_vectors_i;
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
      end else begin
        if (spm_select_valid && spm_req_ready_i) begin
          if (spm_select_write) begin
            spm_valid_q[spm_select_addr] <= 1'b1;
          end else begin
            spm_pending_q <= 1'b1;
            spm_pending_pe_q <= spm_select_pe;
          end
        end
        if (spm_pending_q && spm_rsp_valid_i)
          spm_pending_q <= 1'b0;

        if (running_q) begin
          stat_cycles_q <= stat_cycles_q + 1'b1;
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
          $display("MLX_BACKEND_DONE backend=rtl cycle=%0d instructions=%0d stalls=%0d hops=%0d conflicts=%0d topology=distributed",
                   stat_cycles_q, stat_instructions_q, stat_stall_q,
                   stat_hops_q, stat_conflicts_q);
`endif
        end
      end
    end
  end
endmodule

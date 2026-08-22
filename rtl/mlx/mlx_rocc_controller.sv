`timescale 1ns/1ps

module MLXRoCCBlackBox #(
    parameter BACKEND = 0,
    parameter XLEN = 64,
    parameter ADDR_BITS = 40,
    parameter TAG_BITS = 6,
    parameter CMD_BITS = 5,
    parameter SIZE_BITS = 3,
    parameter DATA_BITS = 64,
    parameter DATA_BYTES = 8
) (
    input  wire                   clock,
    input  wire                   reset,
    output wire                   cmd_ready,
    input  wire                   cmd_valid,
    input  wire [6:0]             cmd_funct,
    input  wire                   cmd_xd,
    input  wire [4:0]             cmd_rd,
    input  wire [XLEN-1:0]        cmd_rs1,
    input  wire [XLEN-1:0]        cmd_rs2,
    input  wire [1:0]             cmd_dprv,
    input  wire                   resp_ready,
    output wire                   resp_valid,
    output wire [4:0]             resp_rd,
    output wire [XLEN-1:0]        resp_data,
    input  wire                   mem_req_ready,
    output wire                   mem_req_valid,
    output wire [ADDR_BITS-1:0]   mem_req_addr,
    output wire [TAG_BITS-1:0]    mem_req_tag,
    output wire [CMD_BITS-1:0]    mem_req_cmd,
    output wire [SIZE_BITS-1:0]   mem_req_size,
    output wire                   mem_req_signed,
    output wire                   mem_req_phys,
    output wire [1:0]             mem_req_dprv,
    output wire [DATA_BITS-1:0]   mem_req_data,
    output wire [DATA_BYTES-1:0]  mem_req_mask,
    input  wire                   mem_resp_valid,
    input  wire [TAG_BITS-1:0]    mem_resp_tag,
    input  wire [DATA_BITS-1:0]   mem_resp_data,
    output wire                   busy
);
  mlx_rocc_controller #(
      .BACKEND(BACKEND),
      .XLEN(XLEN),
      .ADDR_BITS(ADDR_BITS),
      .TAG_BITS(TAG_BITS),
      .CMD_BITS(CMD_BITS),
      .SIZE_BITS(SIZE_BITS),
      .DATA_BITS(DATA_BITS),
      .DATA_BYTES(DATA_BYTES)
  ) controller (
      .clk(clock),
      .rst_n(!reset),
      .cmd_ready_o(cmd_ready),
      .cmd_valid_i(cmd_valid),
      .cmd_funct_i(cmd_funct),
      .cmd_xd_i(cmd_xd),
      .cmd_rd_i(cmd_rd),
      .cmd_rs1_i(cmd_rs1),
      .cmd_rs2_i(cmd_rs2),
      .cmd_dprv_i(cmd_dprv),
      .resp_ready_i(resp_ready),
      .resp_valid_o(resp_valid),
      .resp_rd_o(resp_rd),
      .resp_data_o(resp_data),
      .mem_req_ready_i(mem_req_ready),
      .mem_req_valid_o(mem_req_valid),
      .mem_req_addr_o(mem_req_addr),
      .mem_req_tag_o(mem_req_tag),
      .mem_req_cmd_o(mem_req_cmd),
      .mem_req_size_o(mem_req_size),
      .mem_req_signed_o(mem_req_signed),
      .mem_req_phys_o(mem_req_phys),
      .mem_req_dprv_o(mem_req_dprv),
      .mem_req_data_o(mem_req_data),
      .mem_req_mask_o(mem_req_mask),
      .mem_resp_valid_i(mem_resp_valid),
      .mem_resp_tag_i(mem_resp_tag),
      .mem_resp_data_i(mem_resp_data),
      .busy_o(busy)
  );
endmodule

module mlx_rocc_controller #(
    parameter BACKEND = 0,
    parameter XLEN = 64,
    parameter ADDR_BITS = 40,
    parameter TAG_BITS = 6,
    parameter CMD_BITS = 5,
    parameter SIZE_BITS = 3,
    parameter DATA_BITS = 64,
    parameter DATA_BYTES = 8,
    parameter VECTOR_BITS = 512,
    parameter VECTOR_BEATS = 8,
    parameter SPM_VECTORS = 128
) (
    input  wire                  clk,
    input  wire                  rst_n,
    output reg                   cmd_ready_o,
    input  wire                  cmd_valid_i,
    input  wire [6:0]            cmd_funct_i,
    input  wire                  cmd_xd_i,
    input  wire [4:0]            cmd_rd_i,
    input  wire [XLEN-1:0]       cmd_rs1_i,
    input  wire [XLEN-1:0]       cmd_rs2_i,
    input  wire [1:0]            cmd_dprv_i,
    input  wire                  resp_ready_i,
    output reg                   resp_valid_o,
    output reg [4:0]             resp_rd_o,
    output reg [XLEN-1:0]        resp_data_o,
    input  wire                  mem_req_ready_i,
    output reg                   mem_req_valid_o,
    output reg [ADDR_BITS-1:0]   mem_req_addr_o,
    output wire [TAG_BITS-1:0]   mem_req_tag_o,
    output reg [CMD_BITS-1:0]    mem_req_cmd_o,
    output wire [SIZE_BITS-1:0]  mem_req_size_o,
    output wire                  mem_req_signed_o,
    output wire                  mem_req_phys_o,
    output wire [1:0]            mem_req_dprv_o,
    output reg [DATA_BITS-1:0]   mem_req_data_o,
    output wire [DATA_BYTES-1:0] mem_req_mask_o,
    input  wire                  mem_resp_valid_i,
    input  wire [TAG_BITS-1:0]   mem_resp_tag_i,
    input  wire [DATA_BITS-1:0]  mem_resp_data_i,
    output wire                  busy_o
);
  localparam FUNCT_CONFIG = 7'd0;
  localparam FUNCT_LAUNCH = 7'd1;
  localparam FUNCT_WAIT = 7'd2;
  localparam FUNCT_STATUS = 7'd3;

  localparam ST_IDLE = 4'd0;
  localparam ST_DMA_READ_REQ = 4'd1;
  localparam ST_DMA_READ_RESP = 4'd2;
  localparam ST_BACKEND_START = 4'd3;
  localparam ST_BACKEND_RUN = 4'd4;
  localparam ST_DMA_WRITE_REQ = 4'd5;
  localparam ST_DMA_WRITE_RESP = 4'd6;
  localparam ST_COMPLETE = 4'd7;

  reg [3:0] state_q;
  reg [XLEN-1:0] input_address_q;
  reg [XLEN-1:0] output_address_q;
  reg [1:0] request_dprv_q;
  reg [7:0] input_vectors_q;
  reg [7:0] output_vectors_q;
  reg [7:0] output_base_q;
  reg [7:0] dma_vector_q;
  reg [2:0] dma_beat_q;
  reg [63:0] system_cycles_q;
  reg [63:0] config_commands_q;
  reg [63:0] dma_cycles_q;
  reg [63:0] dma_bytes_q;
  reg [VECTOR_BITS-1:0] spm_q [0:SPM_VECTORS-1];

  wire backend_cfg_valid;
  wire [4:0] backend_cfg_pe;
  wire [5:0] backend_cfg_index;
  wire [63:0] backend_cfg_word;
  wire backend_launch;
  wire backend_spm_req_valid;
  wire backend_spm_req_ready;
  wire backend_spm_req_write;
  wire [7:0] backend_spm_req_addr;
  wire [VECTOR_BITS-1:0] backend_spm_req_wdata;
  reg backend_spm_rsp_valid_q;
  reg [VECTOR_BITS-1:0] backend_spm_rsp_data_q;
  wire backend_busy;
  wire backend_done;
  wire [63:0] backend_stat_cycles;
  wire [63:0] backend_stat_instructions;
  wire [63:0] backend_stat_load;
  wire [63:0] backend_stat_store;
  wire [63:0] backend_stat_compute;
  wire [63:0] backend_stat_xfer;
  wire [63:0] backend_stat_stall;
  wire [63:0] backend_stat_hops;
  wire [63:0] backend_stat_conflicts;

  wire command_fire = cmd_valid_i && cmd_ready_o;
  wire idle_or_complete = (state_q == ST_IDLE) || (state_q == ST_COMPLETE);
  wire [4:0] command_target = cmd_rs2_i[12:8];
  wire [5:0] command_index = cmd_rs2_i[5:0];
  wire unused_command_xd = cmd_xd_i;
  wire [TAG_BITS-1:0] unused_response_tag = mem_resp_tag_i;
  wire [10:0] dma_linear_beat = dma_vector_q * VECTOR_BEATS + dma_beat_q;
  wire [ADDR_BITS-1:0] dma_byte_offset = dma_linear_beat << 3;
  wire [7:0] output_spm_index = output_base_q + dma_vector_q;

  assign backend_cfg_valid = command_fire && (cmd_funct_i == FUNCT_CONFIG)
      && (command_target <= 5'd16);
  assign backend_cfg_pe = command_target;
  assign backend_cfg_index = command_index;
  assign backend_cfg_word = cmd_rs1_i[63:0];
  assign backend_launch = state_q == ST_BACKEND_START;
  assign backend_spm_req_ready = state_q == ST_BACKEND_RUN;

  assign mem_req_tag_o = {TAG_BITS{1'b0}};
  assign mem_req_size_o = {{(SIZE_BITS-2){1'b0}}, 2'd3};
  assign mem_req_signed_o = 1'b0;
  assign mem_req_phys_o = 1'b0;
  assign mem_req_dprv_o = request_dprv_q;
  assign mem_req_mask_o = {DATA_BYTES{1'b1}};
  assign busy_o = !idle_or_complete;

  generate
    if (BACKEND == 0) begin : GENERATE_CYCLE_BACKEND
      mlx_cycle_model backend (
          .clk(clk), .rst_n(rst_n),
          .cfg_valid_i(backend_cfg_valid), .cfg_pe_i(backend_cfg_pe),
          .cfg_index_i(backend_cfg_index), .cfg_word_i(backend_cfg_word),
          .launch_i(backend_launch), .input_vectors_i(input_vectors_q),
          .spm_req_valid_o(backend_spm_req_valid),
          .spm_req_ready_i(backend_spm_req_ready),
          .spm_req_write_o(backend_spm_req_write),
          .spm_req_addr_o(backend_spm_req_addr),
          .spm_req_wdata_o(backend_spm_req_wdata),
          .spm_rsp_valid_i(backend_spm_rsp_valid_q),
          .spm_rsp_rdata_i(backend_spm_rsp_data_q),
          .busy_o(backend_busy), .done_o(backend_done),
          .stat_cycles_o(backend_stat_cycles),
          .stat_instructions_o(backend_stat_instructions),
          .stat_load_o(backend_stat_load), .stat_store_o(backend_stat_store),
          .stat_compute_o(backend_stat_compute), .stat_xfer_o(backend_stat_xfer),
          .stat_stall_o(backend_stat_stall), .stat_hops_o(backend_stat_hops),
          .stat_conflicts_o(backend_stat_conflicts)
      );
    end else begin : GENERATE_RTL_BACKEND
      mlx_array_4x4 backend (
          .clk(clk), .rst_n(rst_n),
          .cfg_valid_i(backend_cfg_valid), .cfg_pe_i(backend_cfg_pe),
          .cfg_index_i(backend_cfg_index), .cfg_word_i(backend_cfg_word),
          .launch_i(backend_launch), .input_vectors_i(input_vectors_q),
          .spm_req_valid_o(backend_spm_req_valid),
          .spm_req_ready_i(backend_spm_req_ready),
          .spm_req_write_o(backend_spm_req_write),
          .spm_req_addr_o(backend_spm_req_addr),
          .spm_req_wdata_o(backend_spm_req_wdata),
          .spm_rsp_valid_i(backend_spm_rsp_valid_q),
          .spm_rsp_rdata_i(backend_spm_rsp_data_q),
          .busy_o(backend_busy), .done_o(backend_done),
          .stat_cycles_o(backend_stat_cycles),
          .stat_instructions_o(backend_stat_instructions),
          .stat_load_o(backend_stat_load), .stat_store_o(backend_stat_store),
          .stat_compute_o(backend_stat_compute), .stat_xfer_o(backend_stat_xfer),
          .stat_stall_o(backend_stat_stall), .stat_hops_o(backend_stat_hops),
          .stat_conflicts_o(backend_stat_conflicts)
      );
    end
  endgenerate

  function [XLEN-1:0] status_value;
    input [3:0] index;
    begin
      case (index)
        4'd0: status_value = {{(XLEN-4){1'b0}}, BACKEND[0],
                              state_q == ST_COMPLETE, busy_o, idle_or_complete};
        4'd1: status_value = system_cycles_q;
        4'd2: status_value = config_commands_q;
        4'd3: status_value = dma_cycles_q;
        4'd4: status_value = backend_stat_cycles;
        4'd5: status_value = backend_stat_instructions;
        4'd6: status_value = backend_stat_load;
        4'd7: status_value = backend_stat_store;
        4'd8: status_value = backend_stat_compute;
        4'd9: status_value = backend_stat_xfer;
        4'd10: status_value = backend_stat_stall;
        4'd11: status_value = backend_stat_hops;
        4'd12: status_value = backend_stat_conflicts;
        4'd13: status_value = dma_bytes_q;
        4'd14: status_value = {{(XLEN-32){1'b0}}, 32'h4d4c5801};
        default: status_value = {XLEN{1'b0}};
      endcase
    end
  endfunction

  always @* begin
    cmd_ready_o = 1'b0;
    resp_valid_o = 1'b0;
    resp_rd_o = cmd_rd_i;
    resp_data_o = {XLEN{1'b0}};
    if (cmd_valid_i) begin
      case (cmd_funct_i)
        FUNCT_CONFIG, FUNCT_LAUNCH: begin
          cmd_ready_o = idle_or_complete;
        end
        FUNCT_WAIT: begin
          resp_valid_o = state_q == ST_COMPLETE;
          resp_data_o = status_value(4'd0);
          cmd_ready_o = (state_q == ST_COMPLETE) && resp_ready_i;
        end
        FUNCT_STATUS: begin
          resp_valid_o = 1'b1;
          resp_data_o = status_value(cmd_rs1_i[3:0]);
          cmd_ready_o = resp_ready_i;
        end
        default: begin
          cmd_ready_o = idle_or_complete;
        end
      endcase
    end

    mem_req_valid_o = 1'b0;
    mem_req_addr_o = {ADDR_BITS{1'b0}};
    mem_req_cmd_o = {CMD_BITS{1'b0}};
    mem_req_data_o = {DATA_BITS{1'b0}};
    if (state_q == ST_DMA_READ_REQ) begin
      mem_req_valid_o = 1'b1;
      mem_req_addr_o = input_address_q[ADDR_BITS-1:0] + dma_byte_offset;
      mem_req_cmd_o = {{(CMD_BITS-1){1'b0}}, 1'b0};
    end else if (state_q == ST_DMA_WRITE_REQ) begin
      mem_req_valid_o = 1'b1;
      mem_req_addr_o = output_address_q[ADDR_BITS-1:0] + dma_byte_offset;
      mem_req_cmd_o = {{(CMD_BITS-1){1'b0}}, 1'b1};
      mem_req_data_o = spm_q[output_spm_index[6:0]]
          [dma_beat_q*DATA_BITS +: DATA_BITS];
    end
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state_q <= ST_IDLE;
      input_address_q <= {XLEN{1'b0}};
      output_address_q <= {XLEN{1'b0}};
      request_dprv_q <= 2'd0;
      input_vectors_q <= 8'd0;
      output_vectors_q <= 8'd0;
      output_base_q <= 8'd64;
      dma_vector_q <= 8'd0;
      dma_beat_q <= 3'd0;
      system_cycles_q <= 64'd0;
      config_commands_q <= 64'd0;
      dma_cycles_q <= 64'd0;
      dma_bytes_q <= 64'd0;
      backend_spm_rsp_valid_q <= 1'b0;
      backend_spm_rsp_data_q <= {VECTOR_BITS{1'b0}};
    end else begin
      backend_spm_rsp_valid_q <= 1'b0;
      if (busy_o)
        system_cycles_q <= system_cycles_q + 1'b1;
      if ((state_q == ST_DMA_READ_REQ) || (state_q == ST_DMA_READ_RESP)
          || (state_q == ST_DMA_WRITE_REQ) || (state_q == ST_DMA_WRITE_RESP))
        dma_cycles_q <= dma_cycles_q + 1'b1;

      if (command_fire && (cmd_funct_i == FUNCT_CONFIG)) begin
        config_commands_q <= config_commands_q + 1'b1;
        if (command_target == 5'd31) begin
          case (command_index)
            6'd0: input_vectors_q <= cmd_rs1_i[7:0];
            6'd1: output_vectors_q <= cmd_rs1_i[7:0];
            6'd2: output_base_q <= cmd_rs1_i[7:0];
            default: begin end
          endcase
        end
      end

      if (command_fire && (cmd_funct_i == FUNCT_LAUNCH)) begin
        input_address_q <= cmd_rs1_i;
        output_address_q <= cmd_rs2_i;
        request_dprv_q <= cmd_dprv_i;
        dma_vector_q <= 8'd0;
        dma_beat_q <= 3'd0;
        system_cycles_q <= 64'd0;
        dma_cycles_q <= 64'd0;
        dma_bytes_q <= 64'd0;
        state_q <= (input_vectors_q == 0) ? ST_BACKEND_START : ST_DMA_READ_REQ;
      end else begin
        case (state_q)
          ST_DMA_READ_REQ: begin
            if (mem_req_valid_o && mem_req_ready_i)
              state_q <= ST_DMA_READ_RESP;
          end
          ST_DMA_READ_RESP: begin
            if (mem_resp_valid_i) begin
              spm_q[dma_vector_q[6:0]][dma_beat_q*DATA_BITS +: DATA_BITS]
                  <= mem_resp_data_i;
              dma_bytes_q <= dma_bytes_q + DATA_BYTES;
              if (dma_beat_q + 1 == VECTOR_BEATS) begin
                dma_beat_q <= 3'd0;
                if (dma_vector_q + 1 == input_vectors_q) begin
                  dma_vector_q <= 8'd0;
                  state_q <= ST_BACKEND_START;
                end else begin
                  dma_vector_q <= dma_vector_q + 1'b1;
                  state_q <= ST_DMA_READ_REQ;
                end
              end else begin
                dma_beat_q <= dma_beat_q + 1'b1;
                state_q <= ST_DMA_READ_REQ;
              end
            end
          end
          ST_BACKEND_START: state_q <= ST_BACKEND_RUN;
          ST_BACKEND_RUN: begin
            if (backend_spm_req_valid && backend_spm_req_ready) begin
              if (backend_spm_req_write) begin
                spm_q[backend_spm_req_addr[6:0]] <= backend_spm_req_wdata;
              end else begin
                backend_spm_rsp_data_q <= spm_q[backend_spm_req_addr[6:0]];
                backend_spm_rsp_valid_q <= 1'b1;
              end
            end
            if (backend_done) begin
              dma_vector_q <= 8'd0;
              dma_beat_q <= 3'd0;
              state_q <= (output_vectors_q == 0) ? ST_COMPLETE : ST_DMA_WRITE_REQ;
            end
          end
          ST_DMA_WRITE_REQ: begin
            if (mem_req_valid_o && mem_req_ready_i)
              state_q <= ST_DMA_WRITE_RESP;
          end
          ST_DMA_WRITE_RESP: begin
            if (mem_resp_valid_i) begin
              dma_bytes_q <= dma_bytes_q + DATA_BYTES;
              if (dma_beat_q + 1 == VECTOR_BEATS) begin
                dma_beat_q <= 3'd0;
                if (dma_vector_q + 1 == output_vectors_q) begin
                  dma_vector_q <= 8'd0;
                  state_q <= ST_COMPLETE;
`ifndef SYNTHESIS
                  $display("MLX_SYSTEM_DONE backend=%0d system_cycles=%0d dma_cycles=%0d backend_cycles=%0d bytes=%0d",
                           BACKEND, system_cycles_q, dma_cycles_q,
                           backend_stat_cycles, dma_bytes_q + DATA_BYTES);
`endif
                end else begin
                  dma_vector_q <= dma_vector_q + 1'b1;
                  state_q <= ST_DMA_WRITE_REQ;
                end
              end else begin
                dma_beat_q <= dma_beat_q + 1'b1;
                state_q <= ST_DMA_WRITE_REQ;
              end
            end
          end
          default: begin end
        endcase
      end
    end
  end
endmodule

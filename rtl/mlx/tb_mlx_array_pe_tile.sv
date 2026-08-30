`timescale 1ns/1ps

module tb_mlx_array_pe_tile;
  reg clk = 1'b0;
  reg rst_n = 1'b0;
  reg cfg_valid = 1'b0;
  reg [4:0] cfg_addr = 5'd0;
  reg [63:0] cfg_word = 64'd0;
  reg cfg_count_valid = 1'b0;
  reg [5:0] cfg_count = 6'd0;
  reg launch = 1'b0;

  wire spm_req_valid;
  wire spm_req_write;
  wire [7:0] spm_req_addr;
  wire [511:0] spm_req_wdata;
  reg spm_rsp_valid = 1'b0;
  reg [511:0] spm_rsp_rdata = 512'd0;

  wire route_out_valid;
  wire [4:0] route_out_target;
  wire signed [4:0] route_out_dx;
  wire signed [4:0] route_out_dy;
  wire [3:0] route_out_destination_register;
  wire [3:0] route_out_tag;
  wire [3:0] route_out_source;
  wire [511:0] route_out_data;
  wire [1:0] route_out_hops;
  wire xfer_complete;
  wire xfer_complete_valid;
  wire [3:0] xfer_complete_source;
  wire execution_active;
  wire router_active;
  wire instruction_issue;
  wire load_issue;
  wire store_issue;
  wire compute_issue;
  wire xfer_issue;
  wire stall;
  wire delivery_conflict;

  reg [511:0] spm [0:3];
  integer index;
  integer timeout;
  integer instruction_count;
  integer load_count;
  integer store_count;
  integer compute_count;
  integer xfer_count;

  always #0.5 clk = ~clk;

  function [63:0] encode;
    input [3:0] opcode;
    input [3:0] tag_id;
    input [1:0] pipeline;
    input [3:0] destination;
    input [3:0] source0;
    input [3:0] source1;
    input [3:0] source2;
    input signed [4:0] dx;
    input signed [4:0] dy;
    input [7:0] immediate;
    begin
      encode = {
          opcode,
          tag_id,
          pipeline,
          destination,
          source0,
          source1,
          source2,
          dx,
          dy,
          immediate,
          20'd0
      };
    end
  endfunction

  assign xfer_complete = xfer_complete_valid
      && (xfer_complete_source == 4'd0);

  mlx_array_pe_tile dut (
      .clk(clk),
      .rst_n(rst_n),
      .cfg_valid_i(cfg_valid),
      .cfg_addr_i(cfg_addr),
      .cfg_word_i(cfg_word),
      .cfg_instruction_count_valid_i(cfg_count_valid),
      .cfg_instruction_count_i(cfg_count),
      .launch_i(launch),
      .tile_id_i(4'd0),
      .spm_req_valid_o(spm_req_valid),
      .spm_req_write_o(spm_req_write),
      .spm_req_addr_o(spm_req_addr),
      .spm_req_wdata_o(spm_req_wdata),
      .spm_req_grant_i(spm_req_valid),
      .spm_rsp_valid_i(spm_rsp_valid),
      .spm_rsp_rdata_i(spm_rsp_rdata),
      .route_in_valid_i(1'b0),
      .route_in_dx_i(5'sd0),
      .route_in_dy_i(5'sd0),
      .route_in_destination_register_i(4'd0),
      .route_in_tag_i(4'd0),
      .route_in_source_i(4'd0),
      .route_in_data_i(512'd0),
      .route_in_ready_o(),
      .route_out_valid_o(route_out_valid),
      .route_out_target_o(route_out_target),
      .route_out_dx_o(route_out_dx),
      .route_out_dy_o(route_out_dy),
      .route_out_destination_register_o(route_out_destination_register),
      .route_out_tag_o(route_out_tag),
      .route_out_source_o(route_out_source),
      .route_out_data_o(route_out_data),
      .route_out_hops_o(route_out_hops),
      .route_out_grant_i(1'b0),
      .xfer_complete_i(xfer_complete),
      .xfer_complete_valid_o(xfer_complete_valid),
      .xfer_complete_source_o(xfer_complete_source),
      .execution_active_o(execution_active),
      .router_active_o(router_active),
      .instruction_issue_o(instruction_issue),
      .load_issue_o(load_issue),
      .store_issue_o(store_issue),
      .compute_issue_o(compute_issue),
      .xfer_issue_o(xfer_issue),
      .stall_o(stall),
      .delivery_conflict_o(delivery_conflict)
  );

  always @(posedge clk) begin
    spm_rsp_valid <= 1'b0;
    if (spm_req_valid) begin
      if (spm_req_write) begin
        spm[spm_req_addr] <= spm_req_wdata;
      end else begin
        spm_rsp_rdata <= spm[spm_req_addr];
        spm_rsp_valid <= 1'b1;
      end
    end
    if (instruction_issue) instruction_count <= instruction_count + 1;
    if (load_issue) load_count <= load_count + 1;
    if (store_issue) store_count <= store_count + 1;
    if (compute_issue) compute_count <= compute_count + 1;
    if (xfer_issue) xfer_count <= xfer_count + 1;
  end

  task configure_word;
    input [4:0] address;
    input [63:0] word;
    begin
      @(negedge clk);
      cfg_valid = 1'b1;
      cfg_addr = address;
      cfg_word = word;
    end
  endtask

  initial begin
    instruction_count = 0;
    load_count = 0;
    store_count = 0;
    compute_count = 0;
    xfer_count = 0;
    for (index = 0; index < 4; index = index + 1)
      spm[index] = 512'd0;
    spm[0] = {32{16'h3c00}};
    spm[1] = {32{16'h3c00}};

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    configure_word(5'd0, encode(4'd0, 4'd0, 2'd0, 4'd0,
                                4'd0, 4'd0, 4'd0, 5'sd0, 5'sd0, 8'd0));
    configure_word(5'd1, encode(4'd0, 4'd0, 2'd0, 4'd1,
                                4'd0, 4'd0, 4'd0, 5'sd0, 5'sd0, 8'd1));
    configure_word(5'd2, encode(4'd3, 4'd0, 2'd2, 4'd2,
                                4'd0, 4'd1, 4'd0, 5'sd0, 5'sd0, 8'd0));
    configure_word(5'd3, encode(4'd8, 4'd0, 2'd3, 4'd3,
                                4'd2, 4'd0, 4'd0, 5'sd0, 5'sd0, 8'd0));
    configure_word(5'd4, encode(4'd1, 4'd0, 2'd1, 4'd0,
                                4'd3, 4'd0, 4'd0, 5'sd0, 5'sd0, 8'd2));

    @(negedge clk);
    cfg_valid = 1'b0;
    cfg_count_valid = 1'b1;
    cfg_count = 6'd5;
    @(negedge clk);
    cfg_count_valid = 1'b0;
    launch = 1'b1;
    @(negedge clk);
    launch = 1'b0;

    timeout = 0;
    while ((store_count == 0) && (timeout < 200)) begin
      @(posedge clk);
      timeout = timeout + 1;
    end
    repeat (2) @(posedge clk);

    if (store_count != 1) begin
      $display("MLX_ARRAY_PE_TILE_TIMEOUT pc=%0d state=%0d instruction=%h rf_valid=%h control_issue=%h tag_active=%h tag_ready=%h tag_done=%h router_valid=%0d router_dx=%0d router_dy=%0d injected=%0d spm_req=%0d rsp=%0d issues=%0d",
               dut.pc_q, dut.state_q, dut.instruction_word, dut.rf_valid_q,
               dut.control_issue_valid, dut.tag_active, dut.tag_ready, dut.tag_done,
               dut.router_valid_q, dut.router_dx_q, dut.router_dy_q,
               dut.xfer_injected_q, spm_req_valid, spm_rsp_valid,
               instruction_count);
      $fatal(1, "tile store did not complete");
    end
    if (spm[2] !== {32{16'h4000}})
      $fatal(1, "tile load/add/xfer/store result mismatch");
    if (instruction_count != 5 || load_count != 2 || compute_count != 1
        || xfer_count != 1)
      $fatal(1, "tile instruction event counts mismatch");
    if (route_out_valid || execution_active || router_active)
      $fatal(1, "tile did not quiesce");
    if (delivery_conflict)
      $fatal(1, "unexpected local delivery conflict");

    $display("MLX_ARRAY_PE_TILE_PASS instructions=%0d loads=%0d compute=%0d xfer=%0d stores=%0d",
             instruction_count, load_count, compute_count, xfer_count, store_count);
    $finish;
  end
endmodule

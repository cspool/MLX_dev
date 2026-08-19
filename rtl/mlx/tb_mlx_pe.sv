`timescale 1ns/1ps

module tb_mlx_pe #(
    parameter SIMD_WIDTH = 32,
    parameter FULL_FEATURES = 1
);
  localparam VECTOR_BITS = SIMD_WIDTH * 16;
  reg clk = 1'b0;
  reg rst_n = 1'b0;
  reg cfg_valid = 1'b0;
  reg [4:0] cfg_addr = 5'd0;
  reg [63:0] cfg_word = 64'd0;
  reg [4:0] fetch_addr = 5'd0;
  wire [63:0] fetch_word;
  reg tag_configure = 1'b0;
  reg [3:0] tag_configure_id = 4'd0;
  reg [7:0] tag_trip_count = 8'd0;
  reg [5:0] tag_frontier = 6'd0;
  reg tag_ready = 1'b0;
  reg tag_issue = 1'b0;
  reg [3:0] tag_issue_id = 4'd0;
  reg tag_complete = 1'b0;
  reg [3:0] tag_complete_id = 4'd0;
  reg [3:0] tag_query_id = 4'd0;
  wire [15:0] tag_active_vector;
  wire [15:0] tag_ready_vector;
  wire [15:0] tag_done_vector;
  wire [7:0] tag_query_trip_count;
  wire [5:0] tag_query_frontier;
  reg [31:0] pipeline_class = 32'd0;
  reg [3:0] pipeline_ready = 4'hf;
  wire [3:0] issue_valid;
  wire [15:0] issue_tag;
  reg [3:0] rf_read_addr_a = 4'd0;
  reg [3:0] rf_read_addr_b = 4'd0;
  wire [VECTOR_BITS-1:0] rf_read_data_a;
  wire [VECTOR_BITS-1:0] rf_read_data_b;
  reg rf_write_enable = 1'b0;
  reg [3:0] rf_write_addr = 4'd0;
  reg [VECTOR_BITS-1:0] rf_write_data = {VECTOR_BITS{1'b0}};
  reg network_valid = 1'b0;
  reg signed [4:0] network_dx = 5'sd0;
  reg signed [4:0] network_dy = 5'sd0;
  wire network_valid_o;
  wire signed [4:0] network_dx_o;
  wire signed [4:0] network_dy_o;
  wire [2:0] network_route_o;
  wire [1:0] network_consumed_hops_o;
  wire network_delivered_o;
  reg fu_valid = 1'b0;
  reg [3:0] fu_op = 4'd0;
  reg [VECTOR_BITS-1:0] vector_a = {VECTOR_BITS{1'b0}};
  reg [VECTOR_BITS-1:0] vector_b = {VECTOR_BITS{1'b0}};
  reg [VECTOR_BITS-1:0] vector_c = {VECTOR_BITS{1'b0}};
  wire fu_valid_o;
  wire [VECTOR_BITS-1:0] vector_result;
  wire fu_illegal;
  reg [63:0] program_mem [0:31];
  reg [1023:0] program_path;
  reg [1023:0] vcd_path;
  reg [1023:0] workload_name;
  integer instruction_count;
  integer instruction_index;
  integer lane;
  integer operation_count;
  integer repetitions;
  integer repeat_index;
  integer activity_mode;
  integer activity_counter;
  reg [31:0] checksum;
  reg [3:0] opcode;
  reg signed [4:0] encoded_dx;
  reg signed [4:0] encoded_dy;
  reg [15:0] expected_lane0;

  mlx_pe_top #(
      .SIMD_WIDTH(SIMD_WIDTH),
      .FULL_FEATURES(FULL_FEATURES)
  ) dut (
      .clk(clk), .rst_n(rst_n),
      .cfg_valid_i(cfg_valid), .cfg_addr_i(cfg_addr), .cfg_word_i(cfg_word),
      .fetch_addr_i(fetch_addr), .fetch_word_o(fetch_word), .configured_o(),
      .tag_configure_i(tag_configure), .tag_configure_id_i(tag_configure_id),
      .tag_trip_count_i(tag_trip_count), .tag_frontier_i(tag_frontier),
      .tag_ready_i(tag_ready), .tag_issue_i(tag_issue), .tag_issue_id_i(tag_issue_id),
      .tag_complete_i(tag_complete), .tag_complete_id_i(tag_complete_id),
      .tag_query_id_i(tag_query_id), .tag_active_vector_o(tag_active_vector),
      .tag_ready_vector_o(tag_ready_vector), .tag_done_vector_o(tag_done_vector),
      .tag_query_trip_count_o(tag_query_trip_count),
      .tag_query_frontier_o(tag_query_frontier),
      .pipeline_class_i(pipeline_class), .pipeline_ready_i(pipeline_ready),
      .issue_valid_o(issue_valid), .issue_tag_o(issue_tag),
      .rf_read_addr_a_i(rf_read_addr_a), .rf_read_addr_b_i(rf_read_addr_b),
      .rf_read_data_a_o(rf_read_data_a), .rf_read_data_b_o(rf_read_data_b),
      .rf_write_enable_i(rf_write_enable), .rf_write_addr_i(rf_write_addr),
      .rf_write_data_i(rf_write_data),
      .network_valid_i(network_valid), .network_dx_i(network_dx),
      .network_dy_i(network_dy), .network_destination_register_i(4'd3),
      .network_tag_i(4'd2), .network_payload_i(64'h123456789abcdef0),
      .network_valid_o(network_valid_o), .network_dx_o(network_dx_o),
      .network_dy_o(network_dy_o), .network_route_o(network_route_o),
      .network_consumed_hops_o(network_consumed_hops_o),
      .network_delivered_o(network_delivered_o), .network_payload_o(),
      .fu_valid_i(fu_valid), .fu_op_i(fu_op), .fu_vector_a_i(vector_a),
      .fu_vector_b_i(vector_b), .fu_vector_c_i(vector_c),
      .fu_valid_o(fu_valid_o), .fu_vector_result_o(vector_result),
      .fu_illegal_o(fu_illegal)
  );

  always #0.5 clk = ~clk;

  task drive_fu;
    input [3:0] selected_opcode;
    begin
      @(negedge clk);
      if (activity_mode != 0) begin
        for (lane = 0; lane < SIMD_WIDTH; lane = lane + 1) begin
          vector_a[lane*16 +: 16] = 16'h3c00 | ((activity_counter + lane) & 1023);
          vector_b[lane*16 +: 16] = 16'h3c00 | ((3*activity_counter + lane) & 1023);
          vector_c[lane*16 +: 16] = 16'h3c00 | ((7*activity_counter + lane) & 1023);
        end
        activity_counter = activity_counter + 1;
      end
      fu_op = selected_opcode;
      fu_valid = 1'b1;
      @(posedge clk);
      #0.01;
      fu_valid = 1'b0;
      if (!fu_valid_o) $fatal(1, "FU result missing");
      if (!FULL_FEATURES && ((selected_opcode == 6) || (selected_opcode == 7))) begin
        if (!fu_illegal) $fatal(1, "reduced operation not rejected");
      end else if (activity_mode == 0) begin
        if (fu_illegal) $fatal(1, "legal FU operation rejected");
        case (selected_opcode)
          2: expected_lane0 = 16'h4500;
          3: expected_lane0 = 16'h4200;
          4: expected_lane0 = 16'h4000;
          5: expected_lane0 = 16'h4170;
          6: expected_lane0 = 16'h3800;
          7: expected_lane0 = 16'h4000;
          9: expected_lane0 = 16'h4000;
          default: expected_lane0 = 16'h3c00;
        endcase
        if (vector_result[15:0] !== expected_lane0)
          $fatal(1, "FP16 lane0 mismatch op=%0d got=%h expected=%h",
                 selected_opcode, vector_result[15:0], expected_lane0);
        checksum = checksum ^ {16'd0, vector_result[15:0]};
      end else begin
        if (fu_illegal) $fatal(1, "activity FU operation rejected");
        checksum = checksum ^ {16'd0, vector_result[15:0]};
      end
    end
  endtask

  initial begin
    if (!$value$plusargs("PROGRAM=%s", program_path)) $fatal(1, "missing PROGRAM");
    if (!$value$plusargs("COUNT=%d", instruction_count)) $fatal(1, "missing COUNT");
    if (!$value$plusargs("WORKLOAD=%s", workload_name)) $fatal(1, "missing WORKLOAD");
    if (!$value$plusargs("VCD=%s", vcd_path)) $fatal(1, "missing VCD");
    if (!$value$plusargs("REPEAT=%d", repetitions)) repetitions = 1;
    if (!$value$plusargs("ACTIVITY=%d", activity_mode)) activity_mode = 0;
    $dumpfile(vcd_path);
    $dumpvars(0, tb_mlx_pe);
    $readmemh(program_path, program_mem);
    checksum = 32'd0;
    operation_count = 0;
    activity_counter = 1;
    for (lane = 0; lane < SIMD_WIDTH; lane = lane + 1) begin
      vector_a[lane*16 +: 16] = (lane == 1) ? 16'h4000 : 16'h3c00;
      vector_b[lane*16 +: 16] = 16'h4000;
      vector_c[lane*16 +: 16] = 16'h4200;
      rf_write_data[lane*16 +: 16] = lane[15:0];
    end
    repeat (3) @(posedge clk);
    rst_n = 1'b1;

    for (instruction_index = 0; instruction_index < instruction_count;
         instruction_index = instruction_index + 1) begin
      @(negedge clk);
      cfg_valid = 1'b1;
      cfg_addr = instruction_index[4:0];
      cfg_word = program_mem[instruction_index];
      @(posedge clk);
    end
    @(negedge clk);
    cfg_valid = 1'b0;
    for (instruction_index = 0; instruction_index < instruction_count;
         instruction_index = instruction_index + 1) begin
      fetch_addr = instruction_index[4:0];
      #0.01;
      if (fetch_word !== program_mem[instruction_index])
        $fatal(1, "config replay mismatch");
    end

    for (instruction_index = 0; instruction_index < 4; instruction_index = instruction_index + 1) begin
      @(negedge clk);
      tag_configure = 1'b1;
      tag_configure_id = instruction_index[3:0];
      tag_trip_count = 8'd4;
      tag_frontier = 6'd0;
      tag_ready = 1'b1;
      pipeline_class[2*instruction_index +: 2] = instruction_index[1:0];
      @(posedge clk);
    end
    @(negedge clk);
    tag_configure = 1'b0;
    #0.01;
    if (issue_valid !== 4'hf) $fatal(1, "four pipelines did not overlap");
    if ((issue_tag[3:0] != 0) || (issue_tag[7:4] != 1)
        || (issue_tag[11:8] != 2) || (issue_tag[15:12] != 3))
      $fatal(1, "lower-tag arbitration mismatch");
    for (instruction_index = 0; instruction_index < 4;
         instruction_index = instruction_index + 1) begin
      @(negedge clk);
      tag_complete = 1'b1;
      tag_complete_id = instruction_index[3:0];
      @(posedge clk);
    end
    @(negedge clk);
    tag_complete = 1'b0;

    @(negedge clk);
    rf_write_enable = 1'b1;
    rf_write_addr = 4'd3;
    @(posedge clk);
    @(negedge clk);
    rf_write_enable = 1'b0;
    rf_read_addr_a = 4'd3;
    #0.01;
    if (rf_read_data_a !== rf_write_data) $fatal(1, "register file mismatch");

    @(negedge clk);
    network_valid = 1'b1;
    network_dx = 5'sd3;
    network_dy = 5'sd0;
    @(posedge clk);
    #0.01;
    if (!network_valid_o || (network_consumed_hops_o != 2) || (network_dx_o != 1))
      $fatal(1, "skip-hop distance-3 first hop mismatch");
    @(negedge clk);
    network_dx = network_dx_o;
    network_dy = network_dy_o;
    @(posedge clk);
    #0.01;
    if ((network_consumed_hops_o != 1) || !network_delivered_o)
      $fatal(1, "skip-hop distance-3 second hop mismatch");
    @(negedge clk);
    network_valid = 1'b0;

    for (repeat_index = 0; repeat_index < repetitions; repeat_index = repeat_index + 1) begin
      for (instruction_index = 0; instruction_index < instruction_count;
           instruction_index = instruction_index + 1) begin
        opcode = program_mem[instruction_index][63:60];
        encoded_dx = program_mem[instruction_index][37:33];
        encoded_dy = program_mem[instruction_index][32:28];
        operation_count = operation_count + 1;
        if ((opcode >= 2) && (opcode <= 7))
          drive_fu(opcode);
        else if (opcode == 9)
          drive_fu(opcode);
        else if (opcode == 8) begin
          @(negedge clk);
          network_valid = 1'b1;
          network_dx = encoded_dx;
          network_dy = encoded_dy;
          @(posedge clk);
          #0.01;
          if (!network_valid_o) $fatal(1, "program xfer did not issue");
          @(negedge clk);
          network_valid = 1'b0;
        end
        if ((activity_mode != 0) && ((opcode == 0) || (opcode == 1)))
          drive_fu(4'd2);
      end
    end
    if (!FULL_FEATURES) drive_fu(4'd6);
    repeat (3) @(posedge clk);
    $display("MLX_RTL_PASS workload=%0s simd=%0d operations=%0d checksum=%08x",
             workload_name, SIMD_WIDTH, operation_count, checksum);
    $finish;
  end
endmodule

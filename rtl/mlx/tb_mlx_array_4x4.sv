`timescale 1ns/1ps

module tb_mlx_array_4x4;
  reg clk = 1'b0;
  reg rst_n = 1'b0;
  reg cfg_valid = 1'b0;
  reg [4:0] cfg_pe = 5'd0;
  reg [5:0] cfg_index = 6'd0;
  reg [63:0] cfg_word = 64'd0;
  reg launch = 1'b0;
  reg [7:0] input_vectors = 8'd0;
  wire spm_req_valid;
  reg spm_req_ready = 1'b1;
  wire spm_req_write;
  wire [7:0] spm_req_addr;
  wire [511:0] spm_req_wdata;
  reg spm_rsp_valid = 1'b0;
  reg [511:0] spm_rsp_rdata = 512'd0;
  wire busy;
  wire done;
  wire [63:0] stat_cycles;
  wire [63:0] stat_instructions;
  wire [63:0] stat_load;
  wire [63:0] stat_store;
  wire [63:0] stat_compute;
  wire [63:0] stat_xfer;
  wire [63:0] stat_stall;
  wire [63:0] stat_hops;
  wire [63:0] stat_conflicts;

  reg [511:0] spm [0:127];
  reg [511:0] golden [0:15];
  reg [4:0] program_pe [0:511];
  reg [5:0] program_index [0:511];
  reg [63:0] program_word [0:511];
  integer instruction_counts [0:15];
  integer program_entries;
  integer output_vectors;
  integer output_base;
  integer program_file;
  integer scan_result;
  integer index;
  integer timeout;
  integer mismatches;
  reg [1023:0] program_path;
  reg [1023:0] input_path;
  reg [1023:0] golden_path;
  reg [1023:0] workload_name;
  reg [1023:0] vcd_path;

  always #0.5 clk = ~clk;

`ifdef MLX_CYCLE_MODEL
  mlx_cycle_model dut (
`elsif MLX_DISTRIBUTED_ARRAY
  mlx_array_4x4_distributed dut (
`else
  mlx_array_4x4 dut (
`endif
      .clk(clk),
      .rst_n(rst_n),
      .cfg_valid_i(cfg_valid),
      .cfg_pe_i(cfg_pe),
      .cfg_index_i(cfg_index),
      .cfg_word_i(cfg_word),
      .launch_i(launch),
      .input_vectors_i(input_vectors),
      .spm_req_valid_o(spm_req_valid),
      .spm_req_ready_i(spm_req_ready),
      .spm_req_write_o(spm_req_write),
      .spm_req_addr_o(spm_req_addr),
      .spm_req_wdata_o(spm_req_wdata),
      .spm_rsp_valid_i(spm_rsp_valid),
      .spm_rsp_rdata_i(spm_rsp_rdata),
      .busy_o(busy),
      .done_o(done),
      .stat_cycles_o(stat_cycles),
      .stat_instructions_o(stat_instructions),
      .stat_load_o(stat_load),
      .stat_store_o(stat_store),
      .stat_compute_o(stat_compute),
      .stat_xfer_o(stat_xfer),
      .stat_stall_o(stat_stall),
      .stat_hops_o(stat_hops),
      .stat_conflicts_o(stat_conflicts)
  );

  always @(posedge clk) begin
    spm_rsp_valid <= 1'b0;
    if (spm_req_valid && spm_req_ready) begin
      if (spm_req_write) begin
        spm[spm_req_addr] <= spm_req_wdata;
      end else begin
        spm_rsp_rdata <= spm[spm_req_addr];
        spm_rsp_valid <= 1'b1;
      end
    end
  end

  initial begin
    if (!$value$plusargs("PROGRAM=%s", program_path)) $fatal(1, "missing PROGRAM");
    if (!$value$plusargs("INPUT=%s", input_path)) $fatal(1, "missing INPUT");
    if (!$value$plusargs("GOLDEN=%s", golden_path)) $fatal(1, "missing GOLDEN");
    if (!$value$plusargs("WORKLOAD=%s", workload_name)) $fatal(1, "missing WORKLOAD");
    if (!$value$plusargs("INPUT_VECTORS=%d", input_vectors))
      $fatal(1, "missing INPUT_VECTORS");
    if (!$value$plusargs("OUTPUT_VECTORS=%d", output_vectors))
      $fatal(1, "missing OUTPUT_VECTORS");
    if (!$value$plusargs("OUTPUT_BASE=%d", output_base)) output_base = 64;
    if ($value$plusargs("VCD=%s", vcd_path)) begin
      $dumpfile(vcd_path);
      $dumpvars(0, tb_mlx_array_4x4);
    end

    for (index = 0; index < 128; index = index + 1)
      spm[index] = 512'd0;
    for (index = 0; index < 16; index = index + 1) begin
      golden[index] = 512'd0;
      instruction_counts[index] = 0;
    end
    $readmemh(input_path, spm);
    $readmemh(golden_path, golden);

    program_entries = 0;
    program_file = $fopen(program_path, "r");
    if (program_file == 0) $fatal(1, "cannot open program");
    while (!$feof(program_file)) begin
      scan_result = $fscanf(program_file, "%x %x %h\n",
                            program_pe[program_entries],
                            program_index[program_entries],
                            program_word[program_entries]);
      if (scan_result == 3) begin
        if (instruction_counts[program_pe[program_entries]]
            < program_index[program_entries] + 1)
          instruction_counts[program_pe[program_entries]]
              = program_index[program_entries] + 1;
        program_entries = program_entries + 1;
      end
    end
    $fclose(program_file);

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);
    for (index = 0; index < program_entries; index = index + 1) begin
      @(negedge clk);
      cfg_valid = 1'b1;
      cfg_pe = program_pe[index];
      cfg_index = program_index[index];
      cfg_word = program_word[index];
    end
    for (index = 0; index < 16; index = index + 1) begin
      @(negedge clk);
      cfg_valid = 1'b1;
      cfg_pe = 5'd16;
      cfg_index = index[5:0];
      cfg_word = instruction_counts[index];
    end
    @(negedge clk);
    cfg_valid = 1'b0;
    launch = 1'b1;
    @(negedge clk);
    launch = 1'b0;

    timeout = 0;
    while (!done && (timeout < 20000)) begin
      @(posedge clk);
      timeout = timeout + 1;
    end
    if (!done) $fatal(1, "array timeout");
    mismatches = 0;
    for (index = 0; index < output_vectors; index = index + 1) begin
      if (spm[output_base + index] !== golden[index]) begin
        $display("MLX_RTL_MISMATCH workload=%0s vector=%0d got=%h expected=%h",
                 workload_name, index, spm[output_base + index], golden[index]);
        mismatches = mismatches + 1;
      end
    end
    if (mismatches != 0) $fatal(1, "array output mismatch");
    $display("MLX_ARRAY_PASS workload=%0s cycles=%0d instructions=%0d load=%0d store=%0d compute=%0d xfer=%0d stalls=%0d hops=%0d conflicts=%0d",
             workload_name, stat_cycles, stat_instructions, stat_load,
             stat_store, stat_compute, stat_xfer, stat_stall,
             stat_hops, stat_conflicts);
    repeat (3) @(posedge clk);
    $finish;
  end
endmodule

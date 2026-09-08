`timescale 1ns/1ps

// Synthesizable control and RF operand front-end of one tagged PE. Functional
// units, SPM and packet network attach through issue/completion/writeback.
// This module is NOT a full execution backend and contains no DPI arithmetic.
module mlx_tagged_pe_control #(
    parameter integer CONTEXTS = 4,
    parameter integer RF_REGISTERS = 16,
    parameter integer PROGRAM_WORDS = 32,
    parameter integer LANES = 32,
    parameter integer VECTOR_BITS = LANES * 16
) (
    input wire clk, rst_n,
    input wire clear_config_i,
    output wire config_ready_o,
    input wire program_write_i,
    input wire [4:0] program_address_i,
    input wire [63:0] program_word_i,
    input wire context_write_i,
    input wire [1:0] context_slot_i,
    input wire [15:0] context_block_i, context_layer_i,
    input wire [7:0] context_epoch_i,
    input wire [4:0] context_start_i,
    input wire [5:0] context_length_i,
    input wire [15:0] context_trips_i,
    input wire [3:0] context_rf_base_i,
    input wire [4:0] context_registers_i,
    input wire admit_i,
    input wire overlap_i,

    // Resource eligibility is distinct from operand readiness. The surrounding
    // PE/array derives it from real FU, SPM and network/receiver availability.
    input wire [CONTEXTS-1:0] resource_ready_i,
    input wire issue_ready_i,
    output reg issue_valid_o,
    output wire issue_fire_o,
    output reg [1:0] issue_slot_o,
    output reg [63:0] issue_word_o,
    output wire [VECTOR_BITS-1:0] issue_operand_a_o, issue_operand_b_o, issue_operand_c_o,

    // Multiple different contexts may complete on one edge, but at most one
    // writeback is accepted. A load/compute completion must own that writeback.
    input wire [CONTEXTS-1:0] complete_i,
    input wire [CONTEXTS*16-1:0] complete_block_i,
    input wire [CONTEXTS*8-1:0] complete_epoch_i,
    input wire [CONTEXTS*16-1:0] complete_iteration_i,
    input wire [CONTEXTS*5-1:0] complete_pc_i,
    input wire writeback_valid_i, writeback_input_i,
    input wire [1:0] writeback_slot_i,
    input wire [15:0] writeback_block_i,
    input wire [7:0] writeback_epoch_i,
    input wire [15:0] writeback_iteration_i,
    input wire [3:0] writeback_register_i,
    input wire [VECTOR_BITS-1:0] writeback_data_i,
    output wire writeback_accept_o,
    output wire [CONTEXTS-1:0] complete_accept_o,
    output reg [CONTEXTS-1:0] retire_o,

    output wire busy_o, done_o,
    output reg error_o,
    output reg [3:0] error_code_o,
    output reg [CONTEXTS-1:0] operand_ready_o,
    output wire [CONTEXTS-1:0] active_o, inflight_o,
    output wire [CONTEXTS*16-1:0] block_o, layer_o, iteration_o, rf_valid_o,
    output wire [CONTEXTS*8-1:0] epoch_o,
    output wire [CONTEXTS*5-1:0] pc_o, registers_o,
    output wire [CONTEXTS*4-1:0] rf_base_o,
    output wire [CONTEXTS*64-1:0] frontier_word_o
);
  reg [63:0] program_q [0:PROGRAM_WORDS-1];
  reg [PROGRAM_WORDS-1:0] program_valid_q;
  reg [CONTEXTS-1:0] configured_q, active_q, inflight_q;
  reg started_q;
  reg [15:0] block_q [0:CONTEXTS-1], layer_q [0:CONTEXTS-1];
  reg [15:0] iteration_q [0:CONTEXTS-1], trips_q [0:CONTEXTS-1];
  reg [15:0] valid_q [0:CONTEXTS-1];
  reg [7:0] epoch_q [0:CONTEXTS-1];
  reg [4:0] pc_q [0:CONTEXTS-1], start_q [0:CONTEXTS-1], registers_q [0:CONTEXTS-1];
  reg [5:0] length_q [0:CONTEXTS-1];
  reg [3:0] rf_base_q [0:CONTEXTS-1];
  reg [VECTOR_BITS-1:0] rf_data_q [0:RF_REGISTERS-1];
  wire [63:0] words [0:CONTEXTS-1];
  reg [3:0] fault;
  integer c, other, address, chosen;
  integer write_slot;
  genvar g;
  initial begin
    if (CONTEXTS < 1 || CONTEXTS > 4 || RF_REGISTERS < 1 || RF_REGISTERS > 16
        || PROGRAM_WORDS < 1 || PROGRAM_WORDS > 32 || LANES < 4 || LANES > 32
        || (LANES & (LANES - 1)) != 0 || VECTOR_BITS != LANES * 16)
      $fatal(1, "unsupported tagged PE front-end geometry");
  end

  function automatic integer arity;
    input [3:0] op;
    begin
      case (op)
        0: arity = 0;
        1, 5, 7, 8: arity = 1;
        2: arity = 3;
        3, 4, 6, 9: arity = 2;
        default: arity = 0;
      endcase
    end
  endfunction
  function automatic writes_local;
    input [3:0] op;
    begin writes_local = op != 1 && op != 8; end
  endfunction
  function automatic legal_word;
    input [63:0] word;
    input [4:0] registers;
    reg [63:0] canonical;
    reg [3:0] op;
    reg [1:0] pipeline;
    integer n;
    begin
      op = word[63:60];
      pipeline = op == 0 ? 0 : op == 1 ? 1 : op == 8 ? 3 : 2;
      n = arity(op);
      canonical = 0;
      canonical[63:60] = op;
      canonical[55:54] = pipeline;
      canonical[53:50] = word[53:50];
      if (n > 0) canonical[49:46] = word[49:46];
      if (n > 1) canonical[45:42] = word[45:42];
      if (n > 2) canonical[41:38] = word[41:38];
      if (op <= 1) canonical[27:12] = word[27:12];
      if (op == 8) begin
        canonical[37:28] = word[37:28];
        canonical[19:4] = word[19:4];
      end
      legal_word = op <= 9 && canonical == word
          && (n == 0 || {1'b0, word[49:46]} < registers)
          && (n <= 1 || {1'b0, word[45:42]} < registers)
          && (n <= 2 || {1'b0, word[41:38]} < registers)
          && (!writes_local(op) || {1'b0, word[53:50]} < registers);
    end
  endfunction

  assign busy_o = |active_q;
  assign done_o = started_q && !busy_o && !error_o;
  assign config_ready_o = !busy_o && !error_o;
  assign active_o = active_q;
  assign inflight_o = inflight_q;
  assign issue_fire_o = issue_valid_o && issue_ready_i;
  assign complete_accept_o = !error_o && fault == 0 ? complete_i : {CONTEXTS{1'b0}};
  assign writeback_accept_o = rst_n && writeback_valid_i && !error_o && fault == 0;
  wire [4:0] read_a = {1'b0, rf_base_q[issue_slot_o]} + {1'b0, issue_word_o[49:46]};
  wire [4:0] read_b = {1'b0, rf_base_q[issue_slot_o]} + {1'b0, issue_word_o[45:42]};
  wire [4:0] read_c = {1'b0, rf_base_q[issue_slot_o]} + {1'b0, issue_word_o[41:38]};
  wire [4:0] write_address = {1'b0, rf_base_q[writeback_slot_i]} + {1'b0, writeback_register_i};
  assign issue_operand_a_o = issue_valid_o && arity(issue_word_o[63:60]) > 0
      && {27'd0, read_a} < RF_REGISTERS ? rf_data_q[read_a[3:0]] : {VECTOR_BITS{1'b0}};
  assign issue_operand_b_o = issue_valid_o && arity(issue_word_o[63:60]) > 1
      && {27'd0, read_b} < RF_REGISTERS ? rf_data_q[read_b[3:0]] : {VECTOR_BITS{1'b0}};
  assign issue_operand_c_o = issue_valid_o && arity(issue_word_o[63:60]) > 2
      && {27'd0, read_c} < RF_REGISTERS ? rf_data_q[read_c[3:0]] : {VECTOR_BITS{1'b0}};
  // Data need not reset: valid_q is cleared at reset/admission/iteration rollover.
  // There is exactly one physical vector write port shared by every context.
  always @(posedge clk)
    if (writeback_accept_o && {27'd0, write_address} < RF_REGISTERS)
      rf_data_q[write_address[3:0]] <= writeback_data_i;
  generate
    for (g = 0; g < CONTEXTS; g = g + 1) begin : context_outputs
      wire [5:0] instruction_address = {1'b0, start_q[g]} + {1'b0, pc_q[g]};
      assign words[g] = {26'd0, instruction_address} < PROGRAM_WORDS
          && program_valid_q[instruction_address[4:0]] ? program_q[instruction_address[4:0]] : 64'd0;
      assign frontier_word_o[g*64 +: 64] = words[g];
      assign block_o[g*16 +: 16] = block_q[g];
      assign layer_o[g*16 +: 16] = layer_q[g];
      assign iteration_o[g*16 +: 16] = iteration_q[g];
      assign epoch_o[g*8 +: 8] = epoch_q[g];
      assign pc_o[g*5 +: 5] = pc_q[g];
      assign rf_valid_o[g*16 +: 16] = valid_q[g];
      assign rf_base_o[g*4 +: 4] = rf_base_q[g];
      assign registers_o[g*5 +: 5] = registers_q[g];
    end
  endgenerate

  always @* begin
    fault = 0;
    write_slot = {30'd0, writeback_slot_i};
    if ((program_write_i || context_write_i || admit_i) && !config_ready_o) fault = 1;
    if (clear_config_i && busy_o) fault = 1;
    if ((clear_config_i && (program_write_i || context_write_i || admit_i))
        || (admit_i && (program_write_i || context_write_i))) fault = 1;
    if (program_write_i && {27'd0, program_address_i} >= PROGRAM_WORDS) fault = 2;
    if (context_write_i && {30'd0, context_slot_i} >= CONTEXTS) fault = 2;
    if (admit_i) begin
      if (configured_q == 0) fault = 2;
      for (c = 0; c < CONTEXTS; c = c + 1) if (configured_q[c]) begin
        if (length_q[c] == 0 || {26'd0, length_q[c]} > PROGRAM_WORDS || trips_q[c] == 0
            || {27'd0, start_q[c]} + {26'd0, length_q[c]} > PROGRAM_WORDS
            || registers_q[c] == 0
            || {28'd0, rf_base_q[c]} + {27'd0, registers_q[c]} > RF_REGISTERS) fault = 2;
        for (other = 0; other < CONTEXTS; other = other + 1)
          if (other != c && configured_q[other]) begin
            if (block_q[c] == block_q[other] || epoch_q[c] != epoch_q[other]) fault = 2;
            if ({1'b0, rf_base_q[c]} < {1'b0, rf_base_q[other]} + registers_q[other]
                && {1'b0, rf_base_q[other]} < {1'b0, rf_base_q[c]} + registers_q[c]) fault = 2;
          end
        for (address = 0; address < PROGRAM_WORDS; address = address + 1)
          if (address >= {27'd0, start_q[c]}
              && address < {27'd0, start_q[c]} + {26'd0, length_q[c]})
            if (!program_valid_q[address] || !legal_word(program_q[address], registers_q[c])) fault = 3;
      end
    end
    for (c = 0; c < CONTEXTS; c = c + 1) if (complete_i[c]) begin
      if (!active_q[c] || !inflight_q[c] || block_q[c] != complete_block_i[c*16 +: 16]
          || epoch_q[c] != complete_epoch_i[c*8 +: 8]
          || iteration_q[c] != complete_iteration_i[c*16 +: 16]
          || pc_q[c] != complete_pc_i[c*5 +: 5]) fault = 4;
      if (writes_local(words[c][63:60]) && (!writeback_valid_i || writeback_input_i
          || write_slot != c || writeback_register_i != words[c][53:50])) fault = 5;
    end
    if (writeback_valid_i) begin
      if (write_slot >= CONTEXTS) fault = 5;
      else begin
        if (!active_q[write_slot] || block_q[write_slot] != writeback_block_i
            || epoch_q[write_slot] != writeback_epoch_i
            || iteration_q[write_slot] != writeback_iteration_i
            || {1'b0, writeback_register_i} >= registers_q[write_slot]) fault = 5;
        if (writeback_input_i) begin
          if (valid_q[write_slot][writeback_register_i]) fault = 5;
        end else if (!complete_i[write_slot] || !writes_local(words[write_slot][63:60])) fault = 5;
      end
    end

    operand_ready_o = 0;
    chosen = -1;
    for (c = 0; c < CONTEXTS; c = c + 1) begin
      if (active_q[c] && !inflight_q[c]) begin
        operand_ready_o[c] = (arity(words[c][63:60]) == 0 || valid_q[c][words[c][49:46]])
            && (arity(words[c][63:60]) <= 1 || valid_q[c][words[c][45:42]])
            && (arity(words[c][63:60]) <= 2 || valid_q[c][words[c][41:38]]);
        if (operand_ready_o[c] && resource_ready_i[c] && (overlap_i || inflight_q == 0)) begin
          if (chosen < 0) chosen = c;
          else if (layer_q[c] < layer_q[chosen]
              || (layer_q[c] == layer_q[chosen] && block_q[c] < block_q[chosen])) chosen = c;
        end
      end
    end
    issue_valid_o = chosen >= 0 && !error_o && fault == 0;
    issue_slot_o = chosen < 0 ? 2'd0 : chosen[1:0];
    issue_word_o = chosen < 0 ? 64'd0 : words[chosen];
    retire_o = 0;
    for (c = 0; c < CONTEXTS; c = c + 1)
      if (complete_accept_o[c] && {1'b0, pc_q[c]} + 6'd1 == length_q[c]
          && {1'b0, iteration_q[c]} + 17'd1 == {1'b0, trips_q[c]}) retire_o[c] = 1;
  end

  integer s;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      configured_q <= 0; active_q <= 0; inflight_q <= 0; program_valid_q <= 0;
      started_q <= 0; error_o <= 0; error_code_o <= 0;
      for (s = 0; s < CONTEXTS; s = s + 1) begin
        block_q[s] <= 0; layer_q[s] <= 0; epoch_q[s] <= 0; iteration_q[s] <= 0;
        trips_q[s] <= 0; pc_q[s] <= 0; start_q[s] <= 0; length_q[s] <= 0;
        rf_base_q[s] <= 0; registers_q[s] <= 0; valid_q[s] <= 0;
      end
    end else if (clear_config_i && !busy_o && !program_write_i && !context_write_i && !admit_i) begin
      configured_q <= 0; active_q <= 0; inflight_q <= 0; program_valid_q <= 0;
      started_q <= 0; error_o <= 0; error_code_o <= 0;
    end else if (!error_o && fault != 0) begin
      error_o <= 1; error_code_o <= fault;
    end else if (!error_o) begin
      if (program_write_i) begin
        program_q[program_address_i] <= program_word_i;
        program_valid_q[program_address_i] <= 1;
      end
      if (context_write_i) begin
        configured_q[context_slot_i] <= 1;
        block_q[context_slot_i] <= context_block_i;
        layer_q[context_slot_i] <= context_layer_i;
        epoch_q[context_slot_i] <= context_epoch_i;
        start_q[context_slot_i] <= context_start_i;
        length_q[context_slot_i] <= context_length_i;
        trips_q[context_slot_i] <= context_trips_i;
        rf_base_q[context_slot_i] <= context_rf_base_i;
        registers_q[context_slot_i] <= context_registers_i;
      end
      if (admit_i) begin
        started_q <= 1; active_q <= configured_q; inflight_q <= 0;
        for (s = 0; s < CONTEXTS; s = s + 1) begin
          pc_q[s] <= 0; iteration_q[s] <= 0; valid_q[s] <= 0;
        end
      end else begin
        if (writeback_accept_o) valid_q[writeback_slot_i][writeback_register_i] <= 1;
        for (s = 0; s < CONTEXTS; s = s + 1) if (complete_accept_o[s]) begin
          inflight_q[s] <= 0;
          if ({1'b0, pc_q[s]} + 6'd1 < length_q[s]) pc_q[s] <= pc_q[s] + 5'd1;
          else if ({1'b0, iteration_q[s]} + 17'd1 < {1'b0, trips_q[s]}) begin
            pc_q[s] <= 0; iteration_q[s] <= iteration_q[s] + 16'd1; valid_q[s] <= 0;
          end else active_q[s] <= 0;
        end
        if (issue_fire_o) inflight_q[issue_slot_o] <= 1;
      end
    end
  end
endmodule

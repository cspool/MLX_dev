`timescale 1ns/1ps

module mlx_register_file #(
    parameter SIMD_WIDTH = 32,
    parameter DATA_BITS = 16,
    parameter DEPTH = 4,
    parameter ADDR_BITS = 4,
    parameter GATED_CLOCK = 1
) (
    input  wire                            clk,
    input  wire                            rst_n,
    input  wire [ADDR_BITS-1:0]            read_addr_a_i,
    input  wire [ADDR_BITS-1:0]            read_addr_b_i,
    input  wire [ADDR_BITS-1:0]            read_addr_c_i,
    output wire [SIMD_WIDTH*DATA_BITS-1:0] read_data_a_o,
    output wire [SIMD_WIDTH*DATA_BITS-1:0] read_data_b_o,
    output wire [SIMD_WIDTH*DATA_BITS-1:0] read_data_c_o,
    input  wire                            write_enable_i,
    input  wire [ADDR_BITS-1:0]            write_addr_i,
    input  wire [SIMD_WIDTH*DATA_BITS-1:0] write_data_i
);
  localparam INDEX_BITS = $clog2(DEPTH);
  reg [SIMD_WIDTH*DATA_BITS-1:0] storage [0:DEPTH-1];
  integer index;
  wire write_clk = clk & write_enable_i;

  assign read_data_a_o = storage[read_addr_a_i[INDEX_BITS-1:0]];
  assign read_data_b_o = storage[read_addr_b_i[INDEX_BITS-1:0]];
  assign read_data_c_o = storage[read_addr_c_i[INDEX_BITS-1:0]];

  generate
    if (GATED_CLOCK != 0) begin : GENERATE_GATED_RF
      always @(posedge write_clk or negedge rst_n) begin
        if (!rst_n) begin
          for (index = 0; index < DEPTH; index = index + 1)
            storage[index] <= {(SIMD_WIDTH*DATA_BITS){1'b0}};
        end else if (write_enable_i) begin
          storage[write_addr_i[INDEX_BITS-1:0]] <= write_data_i;
        end
      end
    end else begin : GENERATE_UNGATED_RF
      always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          for (index = 0; index < DEPTH; index = index + 1)
            storage[index] <= {(SIMD_WIDTH*DATA_BITS){1'b0}};
        end else if (write_enable_i) begin
          storage[write_addr_i[INDEX_BITS-1:0]] <= write_data_i;
        end
      end
    end
  endgenerate
endmodule

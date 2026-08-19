`timescale 1ns/1ps

module mlx_data_network #(
    parameter PAYLOAD_BITS = 64
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    valid_i,
    input  wire signed [4:0]       dx_i,
    input  wire signed [4:0]       dy_i,
    input  wire [3:0]              destination_register_i,
    input  wire [3:0]              tag_i,
    input  wire [PAYLOAD_BITS-1:0] payload_i,
    output wire                    ready_o,
    output reg                     valid_o,
    output reg signed [4:0]        dx_o,
    output reg signed [4:0]        dy_o,
    output reg [3:0]               destination_register_o,
    output reg [3:0]               tag_o,
    output reg [PAYLOAD_BITS-1:0]  payload_o,
    output reg [2:0]               route_o,
    output reg [1:0]               consumed_hops_o,
    output reg                     delivered_o
);
  localparam ROUTE_LOCAL = 3'd0;
  localparam ROUTE_EAST = 3'd1;
  localparam ROUTE_WEST = 3'd2;
  localparam ROUTE_NORTH = 3'd3;
  localparam ROUTE_SOUTH = 3'd4;

  reg signed [4:0] next_dx;
  reg signed [4:0] next_dy;
  reg [2:0] next_route;
  reg [1:0] next_hops;
  reg next_delivered;

  assign ready_o = 1'b1;

  always @* begin
    next_dx = dx_i;
    next_dy = dy_i;
    next_route = ROUTE_LOCAL;
    next_hops = 2'd0;
    next_delivered = (dx_i == 0) && (dy_i == 0);
    if (dx_i > 0) begin
      next_route = ROUTE_EAST;
      next_hops = (dx_i >= 2) ? 2'd2 : 2'd1;
      next_dx = dx_i - $signed({3'd0, next_hops});
      next_delivered = (next_dx == 0) && (dy_i == 0);
    end else if (dx_i < 0) begin
      next_route = ROUTE_WEST;
      next_hops = ((-dx_i) >= 2) ? 2'd2 : 2'd1;
      next_dx = dx_i + $signed({3'd0, next_hops});
      next_delivered = (next_dx == 0) && (dy_i == 0);
    end else if (dy_i > 0) begin
      next_route = ROUTE_NORTH;
      next_hops = (dy_i >= 2) ? 2'd2 : 2'd1;
      next_dy = dy_i - $signed({3'd0, next_hops});
      next_delivered = next_dy == 0;
    end else if (dy_i < 0) begin
      next_route = ROUTE_SOUTH;
      next_hops = ((-dy_i) >= 2) ? 2'd2 : 2'd1;
      next_dy = dy_i + $signed({3'd0, next_hops});
      next_delivered = next_dy == 0;
    end
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      valid_o <= 1'b0;
      dx_o <= 5'sd0;
      dy_o <= 5'sd0;
      destination_register_o <= 4'd0;
      tag_o <= 4'd0;
      payload_o <= {PAYLOAD_BITS{1'b0}};
      route_o <= ROUTE_LOCAL;
      consumed_hops_o <= 2'd0;
      delivered_o <= 1'b0;
    end else begin
      valid_o <= valid_i;
      if (valid_i) begin
        dx_o <= next_dx;
        dy_o <= next_dy;
        destination_register_o <= destination_register_i;
        tag_o <= tag_i;
        payload_o <= payload_i;
        route_o <= next_route;
        consumed_hops_o <= next_hops;
        delivered_o <= next_delivered;
      end
    end
  end
endmodule

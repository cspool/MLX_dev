// Binary16 numerical primitives. No real/shortreal, DPI, or host arithmetic.
// pack rounds an unsigned integer * 2^lsb_exponent to binary16, ties to even.
// These automatic helpers are intentionally local to each module that includes
// them, including a nonlinear child of a lane. There is no shared mutable state.
/* verilator lint_off VARHIDDEN */
function automatic fp_nan;
  input [14:0] value;
  begin fp_nan = value[14:10] == 31 && value[9:0] != 0; end
endfunction


function automatic [63:0] round_shift;
  input [63:0] magnitude;
  input integer shift;
  reg [63:0] retained, low_mask;
  begin
    retained = 0; low_mask = 0;
    if (shift <= 0) retained = magnitude << (-shift);
    else if (shift < 64) begin
      retained = magnitude >> shift;
      low_mask = (64'd1 << (shift - 1)) - 64'd1;
      if (magnitude[shift - 1] && ((magnitude & low_mask) != 0 || retained[0]))
        retained = retained + 64'd1;
    end else if (shift == 64 && magnitude[63] && magnitude[62:0] != 0) retained = 1;
    round_shift = retained;
  end
endfunction

// Balanced leading-bit encoder instead of a 64-deep priority mux chain.
function automatic integer leading_bit;
  input [63:0] magnitude;
  reg [63:0] probe;
  reg [5:0] position;
  begin
    probe = magnitude; position = 0;
    if (probe[63:32] != 0) begin probe = {32'd0, probe[63:32]}; position[5] = 1; end
    if (probe[31:16] != 0) begin probe = {48'd0, probe[31:16]}; position[4] = 1; end
    if (probe[15:8] != 0) begin probe = {56'd0, probe[15:8]}; position[3] = 1; end
    if (probe[7:4] != 0) begin probe = {60'd0, probe[7:4]}; position[2] = 1; end
    if (probe[3:2] != 0) begin probe = {62'd0, probe[3:2]}; position[1] = 1; end
    if (probe[1]) position[0] = 1;
    leading_bit = probe[1:0] == 0 ? -1 : {26'd0, position};
  end
endfunction

function automatic [15:0] fp_pack;
  input sign;
  input [63:0] magnitude;
  input integer lsb_exponent;
  integer high, exponent;
  reg [63:0] rounded;
  reg [4:0] encoded_exponent;
  begin
    high = leading_bit(magnitude); exponent = 0; rounded = 0; encoded_exponent = 0;
    exponent = high + lsb_exponent;
    if (high < 0) fp_pack = {sign, 15'd0};
    else if (exponent > 15) fp_pack = {sign, 5'd31, 10'd0};
    else if (exponent >= -14) begin
      // Normalization guarantees at most 12 bits including rounding carry.
      rounded = round_shift(magnitude, high - 10) & 64'hfff;
      if (rounded >= 2048) begin rounded = rounded >> 1; exponent = exponent + 1; end
      encoded_exponent = exponent[4:0] + 5'd15;
      fp_pack = exponent > 15 ? {sign, 5'd31, 10'd0} : {sign, encoded_exponent, rounded[9:0]};
    end else begin
      rounded = round_shift(magnitude, -24 - lsb_exponent) & 64'h7ff;
      fp_pack = {sign, 4'd0, rounded[10:0]};
    end
  end
endfunction

function automatic [15:0] fp_add;
  input [15:0] a, b;
  integer ea, eb;
  reg [39:0] ma, mb;
  reg [63:0] magnitude;
  reg sign;
  begin
    ea = a[14:10] == 0 ? 1 : {27'd0, a[14:10]};
    eb = b[14:10] == 0 ? 1 : {27'd0, b[14:10]};
    ma = {29'd0, (a[14:10] != 0), a[9:0]} << (ea - 1);
    mb = {29'd0, (b[14:10] != 0), b[9:0]} << (eb - 1);
    magnitude = 0; sign = 0;
    if (a[15] == b[15]) begin
      sign = a[15]; magnitude = {24'd0, ma} + {24'd0, mb};
    end else if (ma >= mb) begin
      sign = a[15]; magnitude = {24'd0, ma} - {24'd0, mb};
    end else begin
      sign = b[15]; magnitude = {24'd0, mb} - {24'd0, ma};
    end
    if (magnitude == 0) sign = a[15] && b[15];
    if (fp_nan(a[14:0]) || fp_nan(b[14:0])) fp_add = 16'h7e00;
    else if (a[14:0] == 15'h7c00 && b[14:0] == 15'h7c00 && a[15] != b[15]) fp_add = 16'hfe00;
    else if (a[14:0] == 15'h7c00) fp_add = a;
    else if (b[14:0] == 15'h7c00) fp_add = b;
    else fp_add = fp_pack(sign, magnitude, -24);
  end
endfunction

function automatic [15:0] fp_mul;
  input [15:0] a, b;
  reg [10:0] ma, mb;
  reg [21:0] product;
  integer ea, eb;
  begin
    ma = {(a[14:10] != 0), a[9:0]}; mb = {(b[14:10] != 0), b[9:0]};
    ea = a[14:10] == 0 ? 1 : {27'd0, a[14:10]};
    eb = b[14:10] == 0 ? 1 : {27'd0, b[14:10]};
    product = ma * mb;
    if (fp_nan(a[14:0]) || fp_nan(b[14:0])) fp_mul = 16'h7e00;
    else if ((a[14:0] == 0 && b[14:0] == 15'h7c00)
        || (b[14:0] == 0 && a[14:0] == 15'h7c00)) fp_mul = 16'hfe00;
    else if (a[14:0] == 15'h7c00 || b[14:0] == 15'h7c00) fp_mul = {a[15] ^ b[15], 5'd31, 10'd0};
    else fp_mul = fp_pack(a[15] ^ b[15], {42'd0, product}, ea + eb - 50);
  end
endfunction

function automatic [15:0] fp_div;
  input [15:0] a, b;
  reg [10:0] ma, mb;
  // An 11-bit significand shifted by 32 requires 43 bits, not a 64-bit divider.
  reg [42:0] numerator, quotient, remainder;
  integer ea, eb;
  begin
    ma = {(a[14:10] != 0), a[9:0]}; mb = {(b[14:10] != 0), b[9:0]};
    ea = a[14:10] == 0 ? 1 : {27'd0, a[14:10]};
    eb = b[14:10] == 0 ? 1 : {27'd0, b[14:10]};
    numerator = {ma, 32'd0}; quotient = 0; remainder = 0;
    if (mb != 0) begin
      quotient = numerator / {32'd0, mb}; remainder = numerator % {32'd0, mb};
      // Carry the exact residual as sticky; at least 11 low quotient bits
      // are discarded by the final binary16 rounding even for tiny mantissas.
      if (remainder != 0) quotient[0] = 1;
    end
    if (fp_nan(a[14:0]) || fp_nan(b[14:0])) fp_div = 16'h7e00;
    else if ((a[14:0] == 0 && b[14:0] == 0)
        || (a[14:0] == 15'h7c00 && b[14:0] == 15'h7c00)) fp_div = 16'hfe00;
    else if (a[14:0] == 15'h7c00 || b[14:0] == 0) fp_div = {a[15] ^ b[15], 5'd31, 10'd0};
    else if (b[14:0] == 15'h7c00 || a[14:0] == 0) fp_div = {a[15] ^ b[15], 15'd0};
    else fp_div = fp_pack(a[15] ^ b[15], {21'd0, quotient}, ea - eb - 32);
  end
endfunction

function automatic [15:0] fp_max;
  input [15:0] a, b;
  begin
    if (fp_nan(a[14:0])) fp_max = a;
    else if (fp_nan(b[14:0])) fp_max = b;
    else if (a[14:0] == 0 && b[14:0] == 0) fp_max = a;
    else if (a[15] != b[15]) fp_max = a[15] ? b : a;
    else if (a[15]) fp_max = a[14:0] <= b[14:0] ? a : b;
    else fp_max = a[14:0] >= b[14:0] ? a : b;
  end
endfunction
/* verilator lint_on VARHIDDEN */

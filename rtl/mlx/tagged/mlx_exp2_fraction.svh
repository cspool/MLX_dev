// Mathematical constants: RNE(2^(j/64) * 2^40), generated with 100-digit Decimal.
// These are range-reduction coefficients, not workload/output lookup values.
function automatic [47:0] exp2_fraction;
  input [5:0] index;
  begin
    case (index)
      6'd0: exp2_fraction = 48'h010000000000;
      6'd1: exp2_fraction = 48'h0102c9a3e778;
      6'd2: exp2_fraction = 48'h01059b0d3158;
      6'd3: exp2_fraction = 48'h01087451875a;
      6'd4: exp2_fraction = 48'h010b5586cf99;
      6'd5: exp2_fraction = 48'h010e3ec32d3d;
      6'd6: exp2_fraction = 48'h0111301d0126;
      6'd7: exp2_fraction = 48'h011429aaea93;
      6'd8: exp2_fraction = 48'h01172b83c7d5;
      6'd9: exp2_fraction = 48'h011a35beb6fd;
      6'd10: exp2_fraction = 48'h011d4873168c;
      6'd11: exp2_fraction = 48'h012063b88629;
      6'd12: exp2_fraction = 48'h012387a6e756;
      6'd13: exp2_fraction = 48'h0126b4565e28;
      6'd14: exp2_fraction = 48'h0129e9df51fe;
      6'd15: exp2_fraction = 48'h012d285a6e40;
      6'd16: exp2_fraction = 48'h01306fe0a31b;
      6'd17: exp2_fraction = 48'h0133c08b2641;
      6'd18: exp2_fraction = 48'h01371a7373ab;
      6'd19: exp2_fraction = 48'h013a7db34e5a;
      6'd20: exp2_fraction = 48'h013dea64c123;
      6'd21: exp2_fraction = 48'h014160a21f73;
      6'd22: exp2_fraction = 48'h0144e0860619;
      6'd23: exp2_fraction = 48'h01486a2b5c14;
      6'd24: exp2_fraction = 48'h014bfdad5363;
      6'd25: exp2_fraction = 48'h014f9b2769d3;
      6'd26: exp2_fraction = 48'h015342b569d5;
      6'd27: exp2_fraction = 48'h0156f4736b52;
      6'd28: exp2_fraction = 48'h015ab07dd485;
      6'd29: exp2_fraction = 48'h015e76f15ad2;
      6'd30: exp2_fraction = 48'h016247eb03a5;
      6'd31: exp2_fraction = 48'h016623882552;
      6'd32: exp2_fraction = 48'h016a09e667f4;
      6'd33: exp2_fraction = 48'h016dfb23c652;
      6'd34: exp2_fraction = 48'h0171f75e8ec6;
      6'd35: exp2_fraction = 48'h0175feb56426;
      6'd36: exp2_fraction = 48'h017a11473eb0;
      6'd37: exp2_fraction = 48'h017e2f336cf5;
      6'd38: exp2_fraction = 48'h0182589994cd;
      6'd39: exp2_fraction = 48'h01868d99b449;
      6'd40: exp2_fraction = 48'h018ace5422aa;
      6'd41: exp2_fraction = 48'h018f1ae99157;
      6'd42: exp2_fraction = 48'h0193737b0cdc;
      6'd43: exp2_fraction = 48'h0197d829fde5;
      6'd44: exp2_fraction = 48'h019c49182a3f;
      6'd45: exp2_fraction = 48'h01a0c667b5de;
      6'd46: exp2_fraction = 48'h01a5503b23e2;
      6'd47: exp2_fraction = 48'h01a9e6b557a0;
      6'd48: exp2_fraction = 48'h01ae89f995ad;
      6'd49: exp2_fraction = 48'h01b33a2b84f1;
      6'd50: exp2_fraction = 48'h01b7f76f2fb6;
      6'd51: exp2_fraction = 48'h01bcc1e904bc;
      6'd52: exp2_fraction = 48'h01c199bdd855;
      6'd53: exp2_fraction = 48'h01c67f12e57d;
      6'd54: exp2_fraction = 48'h01cb720dcef9;
      6'd55: exp2_fraction = 48'h01d072d4a079;
      6'd56: exp2_fraction = 48'h01d5818dcfba;
      6'd57: exp2_fraction = 48'h01da9e603db3;
      6'd58: exp2_fraction = 48'h01dfc97337ba;
      6'd59: exp2_fraction = 48'h01e502ee78b4;
      6'd60: exp2_fraction = 48'h01ea4afa2a49;
      6'd61: exp2_fraction = 48'h01efa1bee616;
      6'd62: exp2_fraction = 48'h01f50765b6e4;
      6'd63: exp2_fraction = 48'h01fa7c1819e9;
    endcase
  end
endfunction

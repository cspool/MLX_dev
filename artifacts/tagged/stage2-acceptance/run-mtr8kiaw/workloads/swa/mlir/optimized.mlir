module attributes {mlx.iterations = 2 : i64, mlx.name = "swa", mlx.numerics = "fp16_step_rounding_nonfused_fma_transcendental_quarter_lanes_v1"} {
  %0 = "mlx.input"() {bits = [46080 : i32, 45948 : i32, 45816 : i32, 45684 : i32, 45551 : i32, 45419 : i32, 45287 : i32, 45155 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], name = "input_0"} : () -> vector<32xf16>
  %1 = "mlx.input"() {bits = [45568 : i32, 45436 : i32, 45304 : i32, 45172 : i32, 45023 : i32, 44759 : i32, 44494 : i32, 44230 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], name = "input_1"} : () -> vector<32xf16>
  %2 = "mlx.input"() {bits = [45056 : i32, 44792 : i32, 44527 : i32, 44263 : i32, 43966 : i32, 43437 : i32, 42810 : i32, 41522 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], name = "input_2"} : () -> vector<32xf16>
  %3 = "mlx.input"() {bits = [44032 : i32, 43503 : i32, 42942 : i32, 41786 : i32, 6177 : i32, 9381 : i32, 10339 : i32, 10868 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], name = "input_3"} : () -> vector<32xf16>
  %4 = "mlx.input"() {bits = [48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32], name = "constant_0"} : () -> vector<32xf16>
  %5 = "mlx.input"() {bits = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], name = "constant_1"} : () -> vector<32xf16>
  %6 = "mlx.compute"(%0, %1) {kind = "mul", layer = 0 : i64, region = "0:dot_0"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %7 = "mlx.compute"(%0, %2) {kind = "mul", layer = 0 : i64, region = "0:dot_1"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %8 = "mlx.compute"(%6, %7) {kind = "max", layer = 1 : i64, region = "1:softmax"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %9 = "mlx.compute"(%8, %4, %6) {kind = "fma", layer = 1 : i64, region = "1:softmax"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %10 = "mlx.compute"(%9) {kind = "exp", layer = 1 : i64, region = "1:softmax"} : (vector<32xf16>) -> vector<32xf16>
  %11 = "mlx.compute"(%8, %4, %7) {kind = "fma", layer = 1 : i64, region = "1:softmax"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %12 = "mlx.compute"(%11) {kind = "exp", layer = 1 : i64, region = "1:softmax"} : (vector<32xf16>) -> vector<32xf16>
  %13 = "mlx.compute"(%10, %12) {kind = "add", layer = 1 : i64, region = "1:softmax"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %14 = "mlx.compute"(%10, %13) {kind = "div", layer = 1 : i64, region = "1:softmax"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %15 = "mlx.compute"(%12, %13) {kind = "div", layer = 1 : i64, region = "1:softmax"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %16 = "mlx.compute"(%14, %2, %5) {kind = "fma", layer = 2 : i64, region = "2:weighted_sum"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %17 = "mlx.compute"(%15, %3, %16) {kind = "fma", layer = 2 : i64, region = "2:weighted_sum"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  "mlx.output"(%17) {name = "v11"} : (vector<32xf16>) -> ()
}


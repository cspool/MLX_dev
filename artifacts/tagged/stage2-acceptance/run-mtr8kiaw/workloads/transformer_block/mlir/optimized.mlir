module attributes {mlx.iterations = 2 : i64, mlx.name = "transformer_block", mlx.numerics = "fp16_step_rounding_nonfused_fma_transcendental_quarter_lanes_v1"} {
  %0 = "mlx.input"() {bits = [46080 : i32, 45948 : i32, 45816 : i32, 45684 : i32, 45551 : i32, 45419 : i32, 45287 : i32, 45155 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], name = "input_0"} : () -> vector<32xf16>
  %1 = "mlx.input"() {bits = [45568 : i32, 45436 : i32, 45304 : i32, 45172 : i32, 45023 : i32, 44759 : i32, 44494 : i32, 44230 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], name = "input_1"} : () -> vector<32xf16>
  %2 = "mlx.input"() {bits = [45056 : i32, 44792 : i32, 44527 : i32, 44263 : i32, 43966 : i32, 43437 : i32, 42810 : i32, 41522 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], name = "input_2"} : () -> vector<32xf16>
  %3 = "mlx.input"() {bits = [44032 : i32, 43503 : i32, 42942 : i32, 41786 : i32, 6177 : i32, 9381 : i32, 10339 : i32, 10868 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], name = "input_3"} : () -> vector<32xf16>
  %4 = "mlx.input"() {bits = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], name = "constant_0"} : () -> vector<32xf16>
  %5 = "mlx.input"() {bits = [48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32, 48128 : i32], name = "constant_1"} : () -> vector<32xf16>
  %6 = "mlx.input"() {bits = [15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32, 15360 : i32], name = "constant_2"} : () -> vector<32xf16>
  %7 = "mlx.input"() {bits = [14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32, 14336 : i32], name = "constant_3"} : () -> vector<32xf16>
  %8 = "mlx.input"() {bits = [47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32, 47104 : i32], name = "constant_4"} : () -> vector<32xf16>
  %9 = "mlx.compute"(%2, %6) {kind = "mul", layer = 0 : i64, region = "0:complex_butterfly_0_1"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %10 = "mlx.compute"(%4, %4, %9) {kind = "fma", layer = 0 : i64, region = "0:complex_butterfly_0_1"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %11 = "mlx.compute"(%0, %10) {kind = "add", layer = 0 : i64, region = "0:complex_butterfly_0_1"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %12 = "mlx.compute"(%10, %5, %0) {kind = "fma", layer = 0 : i64, region = "0:complex_butterfly_0_1"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %13 = "mlx.compute"(%3, %6) {kind = "mul", layer = 0 : i64, region = "0:complex_butterfly_2_3"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %14 = "mlx.compute"(%4, %4, %13) {kind = "fma", layer = 0 : i64, region = "0:complex_butterfly_2_3"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %15 = "mlx.compute"(%3, %4) {kind = "mul", layer = 0 : i64, region = "0:complex_butterfly_2_3"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %16 = "mlx.compute"(%4, %6, %15) {kind = "fma", layer = 0 : i64, region = "0:complex_butterfly_2_3"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %17 = "mlx.compute"(%1, %14) {kind = "add", layer = 0 : i64, region = "0:complex_butterfly_2_3"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %18 = "mlx.compute"(%4, %16) {kind = "add", layer = 0 : i64, region = "0:complex_butterfly_2_3"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %19 = "mlx.compute"(%14, %5, %1) {kind = "fma", layer = 0 : i64, region = "0:complex_butterfly_2_3"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %20 = "mlx.compute"(%16, %5, %4) {kind = "fma", layer = 0 : i64, region = "0:complex_butterfly_2_3"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %21 = "mlx.compute"(%17, %6) {kind = "mul", layer = 1 : i64, region = "1:complex_butterfly_0_2"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %22 = "mlx.compute"(%18, %4, %21) {kind = "fma", layer = 1 : i64, region = "1:complex_butterfly_0_2"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %23 = "mlx.compute"(%11, %22) {kind = "add", layer = 1 : i64, region = "1:complex_butterfly_0_2"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %24 = "mlx.compute"(%22, %5, %11) {kind = "fma", layer = 1 : i64, region = "1:complex_butterfly_0_2"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %25 = "mlx.compute"(%19, %4) {kind = "mul", layer = 1 : i64, region = "1:complex_butterfly_1_3"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %26 = "mlx.compute"(%20, %6, %25) {kind = "fma", layer = 1 : i64, region = "1:complex_butterfly_1_3"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %27 = "mlx.compute"(%12, %26) {kind = "add", layer = 1 : i64, region = "1:complex_butterfly_1_3"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %28 = "mlx.compute"(%26, %5, %12) {kind = "fma", layer = 1 : i64, region = "1:complex_butterfly_1_3"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %29 = "mlx.compute"(%23, %6) {kind = "mul", layer = 2 : i64, region = "2:butterfly_0_1"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %30 = "mlx.compute"(%27, %7, %29) {kind = "fma", layer = 2 : i64, region = "2:butterfly_0_1"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %31 = "mlx.compute"(%23, %8) {kind = "mul", layer = 2 : i64, region = "2:butterfly_0_1"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %32 = "mlx.compute"(%27, %6, %31) {kind = "fma", layer = 2 : i64, region = "2:butterfly_0_1"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %33 = "mlx.compute"(%24, %6) {kind = "mul", layer = 2 : i64, region = "2:butterfly_2_3"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %34 = "mlx.compute"(%28, %7, %33) {kind = "fma", layer = 2 : i64, region = "2:butterfly_2_3"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %35 = "mlx.compute"(%24, %8) {kind = "mul", layer = 2 : i64, region = "2:butterfly_2_3"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %36 = "mlx.compute"(%28, %6, %35) {kind = "fma", layer = 2 : i64, region = "2:butterfly_2_3"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %37 = "mlx.compute"(%30, %6) {kind = "mul", layer = 3 : i64, region = "3:butterfly_0_2"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %38 = "mlx.compute"(%34, %7, %37) {kind = "fma", layer = 3 : i64, region = "3:butterfly_0_2"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %39 = "mlx.compute"(%30, %8) {kind = "mul", layer = 3 : i64, region = "3:butterfly_0_2"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %40 = "mlx.compute"(%34, %6, %39) {kind = "fma", layer = 3 : i64, region = "3:butterfly_0_2"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %41 = "mlx.compute"(%32, %6) {kind = "mul", layer = 3 : i64, region = "3:butterfly_1_3"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %42 = "mlx.compute"(%36, %7, %41) {kind = "fma", layer = 3 : i64, region = "3:butterfly_1_3"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %43 = "mlx.compute"(%32, %8) {kind = "mul", layer = 3 : i64, region = "3:butterfly_1_3"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %44 = "mlx.compute"(%36, %6, %43) {kind = "fma", layer = 3 : i64, region = "3:butterfly_1_3"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %45 = "mlx.compute"(%38, %42) {kind = "mul", layer = 4 : i64, region = "4:dot_0"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %46 = "mlx.compute"(%38, %40) {kind = "mul", layer = 4 : i64, region = "4:dot_1"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %47 = "mlx.compute"(%45, %46) {kind = "max", layer = 5 : i64, region = "5:softmax"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %48 = "mlx.compute"(%47, %5, %45) {kind = "fma", layer = 5 : i64, region = "5:softmax"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %49 = "mlx.compute"(%48) {kind = "exp", layer = 5 : i64, region = "5:softmax"} : (vector<32xf16>) -> vector<32xf16>
  %50 = "mlx.compute"(%47, %5, %46) {kind = "fma", layer = 5 : i64, region = "5:softmax"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %51 = "mlx.compute"(%50) {kind = "exp", layer = 5 : i64, region = "5:softmax"} : (vector<32xf16>) -> vector<32xf16>
  %52 = "mlx.compute"(%49, %51) {kind = "add", layer = 5 : i64, region = "5:softmax"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %53 = "mlx.compute"(%49, %52) {kind = "div", layer = 5 : i64, region = "5:softmax"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %54 = "mlx.compute"(%51, %52) {kind = "div", layer = 5 : i64, region = "5:softmax"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %55 = "mlx.compute"(%53, %40, %4) {kind = "fma", layer = 6 : i64, region = "6:weighted_sum"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %56 = "mlx.compute"(%54, %44, %55) {kind = "fma", layer = 6 : i64, region = "6:weighted_sum"} : (vector<32xf16>, vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  %57 = "mlx.compute"(%56, %0) {kind = "add", layer = 7 : i64, region = "7:residual"} : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
  "mlx.output"(%57) {name = "v60"} : (vector<32xf16>) -> ()
}


module @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D {
  func.func @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D(%input: tensor<128x32x112x112xf32>, %filter: tensor<32x32x3x3xf32>) -> tensor<128x32x112x112xf32> {
    %result = "stablehlo.convolution"(%input, %filter) {
        window_strides = dense<[1, 1]> : tensor<2xi64>,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        lhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>,
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64}
        : (tensor<128x32x112x112xf32>, tensor<32x32x3x3xf32>) -> tensor<128x32x112x112xf32>
    return %result : tensor<128x32x112x112xf32>
  }
}

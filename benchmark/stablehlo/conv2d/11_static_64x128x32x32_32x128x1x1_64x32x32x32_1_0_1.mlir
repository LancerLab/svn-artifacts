module @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1 {
  func.func @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1(%input: tensor<64x128x32x32xf32>, %filter: tensor<32x128x1x1xf32>) -> tensor<64x32x32x32xf32> {
    %result = "stablehlo.convolution"(%input, %filter) {
        window_strides = dense<[1, 1]> : tensor<2xi64>,
        padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>,
        lhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>,
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64}
        : (tensor<64x128x32x32xf32>, tensor<32x128x1x1xf32>) -> tensor<64x32x32x32xf32>
    return %result : tensor<64x32x32x32xf32>
  }
}

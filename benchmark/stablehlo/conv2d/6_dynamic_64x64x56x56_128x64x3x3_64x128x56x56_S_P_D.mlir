module @f_6_dynamic_64x64x56x56_128x64x3x3_64x128x56x56_S_P_D {
  func.func @f_6_dynamic_64x64x56x56_128x64x3x3_64x128x56x56_S_P_D(%input: tensor<64x64x56x56xf32>, %filter: tensor<128x64x3x3xf32>) -> tensor<64x128x56x56xf32> {
    %result = "stablehlo.convolution"(%input, %filter) {
        window_strides = dense<[1, 1]> : tensor<2xi64>,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        lhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>,
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64}
        : (tensor<64x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<64x128x56x56xf32>
    return %result : tensor<64x128x56x56xf32>
  }
}

module @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D {
  func.func @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D(%input: tensor<32x128x112x112xf32>, %filter: tensor<256x128x3x3xf32>) -> tensor<32x256x56x56xf32> {
    %result = "stablehlo.convolution"(%input, %filter) {
        window_strides = dense<[2, 2]> : tensor<2xi64>,
        padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
        lhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>,
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64}
        : (tensor<32x128x112x112xf32>, tensor<256x128x3x3xf32>) -> tensor<32x256x56x56xf32>
    return %result : tensor<32x256x56x56xf32>
  }
}

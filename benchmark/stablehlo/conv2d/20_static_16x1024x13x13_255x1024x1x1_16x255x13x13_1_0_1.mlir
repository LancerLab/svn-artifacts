module @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1 {
  func.func @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1(%input: tensor<16x1024x13x13xf32>, %filter: tensor<255x1024x1x1xf32>) -> tensor<16x255x13x13xf32> {
    %result = "stablehlo.convolution"(%input, %filter) {
        window_strides = dense<[1, 1]> : tensor<2xi64>,
        padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>,
        lhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>,
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64}
        : (tensor<16x1024x13x13xf32>, tensor<255x1024x1x1xf32>) -> tensor<16x255x13x13xf32>
    return %result : tensor<16x255x13x13xf32>
  }
}

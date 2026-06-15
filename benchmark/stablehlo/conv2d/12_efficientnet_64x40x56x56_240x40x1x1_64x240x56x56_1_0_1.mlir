module @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1 {
  func.func @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1(%input: tensor<64x40x56x56xf32>, %filter: tensor<240x40x1x1xf32>) -> tensor<64x240x56x56xf32> {
    %result = "stablehlo.convolution"(%input, %filter) {
        window_strides = dense<[1, 1]> : tensor<2xi64>,
        padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>,
        lhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>,
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64}
        : (tensor<64x40x56x56xf32>, tensor<240x40x1x1xf32>) -> tensor<64x240x56x56xf32>
    return %result : tensor<64x240x56x56xf32>
  }
}

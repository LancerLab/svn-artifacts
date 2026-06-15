module @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1 {
  func.func @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1(%input: tensor<32x192x28x28xf32>, %filter: tensor<64x192x1x1xf32>) -> tensor<32x64x28x28xf32> {
    %result = "stablehlo.convolution"(%input, %filter) {
        window_strides = dense<[1, 1]> : tensor<2xi64>,
        padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>,
        lhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>,
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64}
        : (tensor<32x192x28x28xf32>, tensor<64x192x1x1xf32>) -> tensor<32x64x28x28xf32>
    return %result : tensor<32x64x28x28xf32>
  }
}

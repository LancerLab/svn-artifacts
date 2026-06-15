module @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1 {
  func.func @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1(%input: tensor<8x256x64x64xf32>, %filter: tensor<21x256x1x1xf32>) -> tensor<8x21x64x64xf32> {
    %result = "stablehlo.convolution"(%input, %filter) {
        window_strides = dense<[1, 1]> : tensor<2xi64>,
        padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>,
        lhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>,
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64}
        : (tensor<8x256x64x64xf32>, tensor<21x256x1x1xf32>) -> tensor<8x21x64x64xf32>
    return %result : tensor<8x21x64x64xf32>
  }
}

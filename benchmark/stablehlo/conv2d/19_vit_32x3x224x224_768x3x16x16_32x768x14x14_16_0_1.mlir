module @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1 {
  func.func @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1(%input: tensor<32x3x224x224xf32>, %filter: tensor<768x3x16x16xf32>) -> tensor<32x768x14x14xf32> {
    %result = "stablehlo.convolution"(%input, %filter) {
        window_strides = dense<[16, 16]> : tensor<2xi64>,
        padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>,
        lhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>,
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64}
        : (tensor<32x3x224x224xf32>, tensor<768x3x16x16xf32>) -> tensor<32x768x14x14xf32>
    return %result : tensor<32x768x14x14xf32>
  }
}

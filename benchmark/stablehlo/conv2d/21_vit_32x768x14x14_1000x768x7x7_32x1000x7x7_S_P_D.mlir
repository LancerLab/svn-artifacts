module @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D {
  func.func @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D(%input: tensor<32x768x14x14xf32>, %filter: tensor<1000x768x7x7xf32>) -> tensor<32x1000x7x7xf32> {
    %result = "stablehlo.convolution"(%input, %filter) {
        window_strides = dense<[2, 2]> : tensor<2xi64>,
        padding = dense<[[3, 3], [3, 3]]> : tensor<2x2xi64>,
        lhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>,
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64}
        : (tensor<32x768x14x14xf32>, tensor<1000x768x7x7xf32>) -> tensor<32x1000x7x7xf32>
    return %result : tensor<32x1000x7x7xf32>
  }
}

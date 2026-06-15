module @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D {
  func.func @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D(%input: tensor<64x3x224x224xf32>, %filter: tensor<64x3x7x7xf32>) -> tensor<64x64x112x112xf32> {
    %result = "stablehlo.convolution"(%input, %filter) {
        window_strides = dense<[2, 2]> : tensor<2xi64>,
        padding = dense<[[3, 3], [3, 3]]> : tensor<2x2xi64>,
        lhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
        dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>,
        feature_group_count = 1 : i64,
        batch_group_count = 1 : i64}
        : (tensor<64x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<64x64x112x112xf32>
    return %result : tensor<64x64x112x112xf32>
  }
}

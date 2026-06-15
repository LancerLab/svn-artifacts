module @f_14_efficientnet_64x1280x7x7_1280_1280_64x1280x7x7 {
  func.func @f_14_efficientnet_64x1280x7x7_1280_1280_64x1280x7x7(%input: tensor<64x1280x7x7xf32>, %gamma: tensor<1280xf32>, %beta: tensor<1280xf32>) -> tensor<64x1280x7x7xf32> {
    %mean    = stablehlo.constant dense<0.0> : tensor<1280xf32>
    %var     = stablehlo.constant dense<1.0> : tensor<1280xf32>
    %result  = "stablehlo.batch_norm_inference"(%input, %gamma, %beta, %mean, %var)
        {epsilon = 1.000000e-05 : f32, feature_index = 1 : i64}
        : (tensor<64x1280x7x7xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>) -> tensor<64x1280x7x7xf32>
    return %result : tensor<64x1280x7x7xf32>
  }
}

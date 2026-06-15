module @f_19_transformer_32x512x2048_2048_2048_32x512x2048 {
  func.func @f_19_transformer_32x512x2048_2048_2048_32x512x2048(%input: tensor<32x512x2048xf32>, %gamma: tensor<2048xf32>, %beta: tensor<2048xf32>) -> tensor<32x512x2048xf32> {
    %mean    = stablehlo.constant dense<0.0> : tensor<2048xf32>
    %var     = stablehlo.constant dense<1.0> : tensor<2048xf32>
    %result  = "stablehlo.batch_norm_inference"(%input, %gamma, %beta, %mean, %var)
        {epsilon = 1.000000e-05 : f32, feature_index = 2 : i64}
        : (tensor<32x512x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<32x512x2048xf32>
    return %result : tensor<32x512x2048xf32>
  }
}

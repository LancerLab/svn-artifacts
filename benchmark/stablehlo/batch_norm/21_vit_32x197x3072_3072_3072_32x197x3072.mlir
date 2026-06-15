module @f_21_vit_32x197x3072_3072_3072_32x197x3072 {
  func.func @f_21_vit_32x197x3072_3072_3072_32x197x3072(%input: tensor<32x197x3072xf32>, %gamma: tensor<3072xf32>, %beta: tensor<3072xf32>) -> tensor<32x197x3072xf32> {
    %mean    = stablehlo.constant dense<0.0> : tensor<3072xf32>
    %var     = stablehlo.constant dense<1.0> : tensor<3072xf32>
    %result  = "stablehlo.batch_norm_inference"(%input, %gamma, %beta, %mean, %var)
        {epsilon = 1.000000e-05 : f32, feature_index = 2 : i64}
        : (tensor<32x197x3072xf32>, tensor<3072xf32>, tensor<3072xf32>, tensor<3072xf32>, tensor<3072xf32>) -> tensor<32x197x3072xf32>
    return %result : tensor<32x197x3072xf32>
  }
}

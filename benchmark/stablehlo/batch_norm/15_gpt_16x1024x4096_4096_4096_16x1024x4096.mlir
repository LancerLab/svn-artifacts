module @f_15_gpt_16x1024x4096_4096_4096_16x1024x4096 {
  func.func @f_15_gpt_16x1024x4096_4096_4096_16x1024x4096(%input: tensor<16x1024x4096xf32>, %gamma: tensor<4096xf32>, %beta: tensor<4096xf32>) -> tensor<16x1024x4096xf32> {
    %mean    = stablehlo.constant dense<0.0> : tensor<4096xf32>
    %var     = stablehlo.constant dense<1.0> : tensor<4096xf32>
    %result  = "stablehlo.batch_norm_inference"(%input, %gamma, %beta, %mean, %var)
        {epsilon = 1.000000e-05 : f32, feature_index = 2 : i64}
        : (tensor<16x1024x4096xf32>, tensor<4096xf32>, tensor<4096xf32>, tensor<4096xf32>, tensor<4096xf32>) -> tensor<16x1024x4096xf32>
    return %result : tensor<16x1024x4096xf32>
  }
}

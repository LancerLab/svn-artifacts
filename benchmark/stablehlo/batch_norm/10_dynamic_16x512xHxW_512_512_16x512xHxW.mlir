module @f_10_dynamic_16x512xHxW_512_512_16x512xHxW {
  func.func @f_10_dynamic_16x512xHxW_512_512_16x512xHxW(%input: tensor<16x512x?x?xf32>, %gamma: tensor<512xf32>, %beta: tensor<512xf32>) -> tensor<16x512x?x?xf32> {
    %mean    = stablehlo.constant dense<0.0> : tensor<512xf32>
    %var     = stablehlo.constant dense<1.0> : tensor<512xf32>
    %result  = "stablehlo.batch_norm_inference"(%input, %gamma, %beta, %mean, %var)
        {epsilon = 1.000000e-05 : f32, feature_index = 1 : i64}
        : (tensor<16x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<16x512x?x?xf32>
    return %result : tensor<16x512x?x?xf32>
  }
}

module @f_9_dynamic_64x128xHxW_128_128_64x128xHxW {
  func.func @f_9_dynamic_64x128xHxW_128_128_64x128xHxW(%input: tensor<64x128x?x?xf32>, %gamma: tensor<128xf32>, %beta: tensor<128xf32>) -> tensor<64x128x?x?xf32> {
    %mean    = stablehlo.constant dense<0.0> : tensor<128xf32>
    %var     = stablehlo.constant dense<1.0> : tensor<128xf32>
    %result  = "stablehlo.batch_norm_inference"(%input, %gamma, %beta, %mean, %var)
        {epsilon = 1.000000e-05 : f32, feature_index = 1 : i64}
        : (tensor<64x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<64x128x?x?xf32>
    return %result : tensor<64x128x?x?xf32>
  }
}

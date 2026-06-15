module @f_20_unet_16x512x32x32_512_512_16x512x32x32 {
  func.func @f_20_unet_16x512x32x32_512_512_16x512x32x32(%input: tensor<16x512x32x32xf32>, %gamma: tensor<512xf32>, %beta: tensor<512xf32>) -> tensor<16x512x32x32xf32> {
    %mean    = stablehlo.constant dense<0.0> : tensor<512xf32>
    %var     = stablehlo.constant dense<1.0> : tensor<512xf32>
    %result  = "stablehlo.batch_norm_inference"(%input, %gamma, %beta, %mean, %var)
        {epsilon = 1.000000e-05 : f32, feature_index = 1 : i64}
        : (tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<16x512x32x32xf32>
    return %result : tensor<16x512x32x32xf32>
  }
}

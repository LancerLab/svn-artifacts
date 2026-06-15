module @f_2_cnn_128x128x28x28_128_128_128x128x28x28 {
  func.func @f_2_cnn_128x128x28x28_128_128_128x128x28x28(%input: tensor<128x128x28x28xf32>, %gamma: tensor<128xf32>, %beta: tensor<128xf32>) -> tensor<128x128x28x28xf32> {
    %mean    = stablehlo.constant dense<0.0> : tensor<128xf32>
    %var     = stablehlo.constant dense<1.0> : tensor<128xf32>
    %result  = "stablehlo.batch_norm_inference"(%input, %gamma, %beta, %mean, %var)
        {epsilon = 1.000000e-05 : f32, feature_index = 0 : i64}
        : (tensor<128x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<128x128x28x28xf32>
    return %result : tensor<128x128x28x28xf32>
  }
}

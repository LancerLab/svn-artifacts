module @f_17_mobilenet_128x96x112x112_96_96_128x96x112x112 {
  func.func @f_17_mobilenet_128x96x112x112_96_96_128x96x112x112(%input: tensor<128x96x112x112xf32>, %gamma: tensor<96xf32>, %beta: tensor<96xf32>) -> tensor<128x96x112x112xf32> {
    %mean    = stablehlo.constant dense<0.0> : tensor<96xf32>
    %var     = stablehlo.constant dense<1.0> : tensor<96xf32>
    %result  = "stablehlo.batch_norm_inference"(%input, %gamma, %beta, %mean, %var)
        {epsilon = 1.000000e-05 : f32, feature_index = 1 : i64}
        : (tensor<128x96x112x112xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<128x96x112x112xf32>
    return %result : tensor<128x96x112x112xf32>
  }
}

module @f_11_dynamic_32xSx768_768_768_32xSx768 {
  func.func @f_11_dynamic_32xSx768_768_768_32xSx768(%input: tensor<32x?x768xf32>, %gamma: tensor<768xf32>, %beta: tensor<768xf32>) -> tensor<32x?x768xf32> {
    %mean    = stablehlo.constant dense<0.0> : tensor<768xf32>
    %var     = stablehlo.constant dense<1.0> : tensor<768xf32>
    %result  = "stablehlo.batch_norm_inference"(%input, %gamma, %beta, %mean, %var)
        {epsilon = 1.000000e-05 : f32, feature_index = 2 : i64}
        : (tensor<32x?x768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<32x?x768xf32>
    return %result : tensor<32x?x768xf32>
  }
}

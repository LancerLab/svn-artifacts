module @f_5_dynamic_Nx1280xHxW_1280_1280_Nx1280xHxW {
  func.func @f_5_dynamic_Nx1280xHxW_1280_1280_Nx1280xHxW(%input: tensor<?x1280x?x?xf32>, %gamma: tensor<1280xf32>, %beta: tensor<1280xf32>) -> tensor<?x1280x?x?xf32> {
    %mean    = stablehlo.constant dense<0.0> : tensor<1280xf32>
    %var     = stablehlo.constant dense<1.0> : tensor<1280xf32>
    %result  = "stablehlo.batch_norm_inference"(%input, %gamma, %beta, %mean, %var)
        {epsilon = 1.000000e-05 : f32, feature_index = 1 : i64}
        : (tensor<?x1280x?x?xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>) -> tensor<?x1280x?x?xf32>
    return %result : tensor<?x1280x?x?xf32>
  }
}

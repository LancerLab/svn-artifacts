module @f_4_dynamic_Nx256x56x56_256_256_Nx256x56x56 {
  func.func @f_4_dynamic_Nx256x56x56_256_256_Nx256x56x56(%input: tensor<?x256x56x56xf32>, %gamma: tensor<256xf32>, %beta: tensor<256xf32>) -> tensor<?x256x56x56xf32> {
    %mean    = stablehlo.constant dense<0.0> : tensor<256xf32>
    %var     = stablehlo.constant dense<1.0> : tensor<256xf32>
    %result  = "stablehlo.batch_norm_inference"(%input, %gamma, %beta, %mean, %var)
        {epsilon = 1.000000e-05 : f32, feature_index = 1 : i64}
        : (tensor<?x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<?x256x56x56xf32>
    return %result : tensor<?x256x56x56xf32>
  }
}

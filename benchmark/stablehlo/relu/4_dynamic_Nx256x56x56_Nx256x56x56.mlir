module @f_4_dynamic_Nx256x56x56_Nx256x56x56 {
  func.func @f_4_dynamic_Nx256x56x56_Nx256x56x56(%input: tensor<?x256x56x56xf32>) -> tensor<?x256x56x56xf32> {
    %zero   = stablehlo.constant dense<0.0> : tensor<f32>
    %inf    = stablehlo.constant dense<3.4028235e+38> : tensor<f32>
    %result = stablehlo.clamp %zero, %input, %inf : (tensor<f32>, tensor<?x256x56x56xf32>, tensor<f32>) -> tensor<?x256x56x56xf32>
    return %result : tensor<?x256x56x56xf32>
  }
}

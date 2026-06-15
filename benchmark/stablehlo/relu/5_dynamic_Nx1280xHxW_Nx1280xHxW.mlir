module @f_5_dynamic_Nx1280xHxW_Nx1280xHxW {
  func.func @f_5_dynamic_Nx1280xHxW_Nx1280xHxW(%input: tensor<?x1280x?x?xf32>) -> tensor<?x1280x?x?xf32> {
    %zero   = stablehlo.constant dense<0.0> : tensor<f32>
    %inf    = stablehlo.constant dense<3.4028235e+38> : tensor<f32>
    %result = stablehlo.clamp %zero, %input, %inf : (tensor<f32>, tensor<?x1280x?x?xf32>, tensor<f32>) -> tensor<?x1280x?x?xf32>
    return %result : tensor<?x1280x?x?xf32>
  }
}

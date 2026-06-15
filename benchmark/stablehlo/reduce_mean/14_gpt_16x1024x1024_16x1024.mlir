module @f_14_gpt_16x1024x1024_16x1024 {
  func.func @f_14_gpt_16x1024x1024_16x1024(%input: tensor<16x1024x1024xf32>) -> tensor<16x1024xf32> {
    %zero     = stablehlo.constant dense<0.0> : tensor<f32>
    %sum_red  = stablehlo.reduce(%input init: %zero) across dimensions = [2] : (tensor<16x1024x1024xf32>, tensor<f32>) -> tensor<16x1024xf32>
      reducer(%lhs: tensor<f32>, %rhs: tensor<f32>)  {
        %s = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %s : tensor<f32>
      }
    %nsz      = stablehlo.constant dense<1024.0> : tensor<16x1024xf32>
    %result   = stablehlo.divide %sum_red, %nsz : tensor<16x1024xf32>
    return %result : tensor<16x1024xf32>
  }
}

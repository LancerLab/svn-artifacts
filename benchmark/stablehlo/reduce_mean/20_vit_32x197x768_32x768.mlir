module @f_20_vit_32x197x768_32x768 {
  func.func @f_20_vit_32x197x768_32x768(%input: tensor<32x197x768xf32>) -> tensor<32x768xf32> {
    %zero     = stablehlo.constant dense<0.0> : tensor<f32>
    %sum_red  = stablehlo.reduce(%input init: %zero) across dimensions = [1] : (tensor<32x197x768xf32>, tensor<f32>) -> tensor<32x768xf32>
      reducer(%lhs: tensor<f32>, %rhs: tensor<f32>)  {
        %s = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %s : tensor<f32>
      }
    %nsz      = stablehlo.constant dense<197.0> : tensor<32x768xf32>
    %result   = stablehlo.divide %sum_red, %nsz : tensor<32x768xf32>
    return %result : tensor<32x768xf32>
  }
}

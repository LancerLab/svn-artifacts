module @f_19_unet_16x512x32x32_16x512x1x1 {
  func.func @f_19_unet_16x512x32x32_16x512x1x1(%input: tensor<16x512x32x32xf32>) -> tensor<16x512xf32> {
    %zero     = stablehlo.constant dense<0.0> : tensor<f32>
    %sum_red  = stablehlo.reduce(%input init: %zero) across dimensions = [2, 3] : (tensor<16x512x32x32xf32>, tensor<f32>) -> tensor<16x512xf32>
      reducer(%lhs: tensor<f32>, %rhs: tensor<f32>)  {
        %s = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %s : tensor<f32>
      }
    %nsz      = stablehlo.constant dense<1024.0> : tensor<16x512xf32>
    %result   = stablehlo.divide %sum_red, %nsz : tensor<16x512xf32>
    return %result : tensor<16x512xf32>
  }
}

module @f_2_cnn_128x128x28x28_128x128x28x28 {
  func.func @f_2_cnn_128x128x28x28_128x128x28x28(%input: tensor<128x128x28x28xf32>) -> tensor<128x128x28x28xf32> {
    %zero   = stablehlo.constant dense<0.0> : tensor<f32>
    %inf    = stablehlo.constant dense<3.4028235e+38> : tensor<f32>
    %result = stablehlo.clamp %zero, %input, %inf : (tensor<f32>, tensor<128x128x28x28xf32>, tensor<f32>) -> tensor<128x128x28x28xf32>
    return %result : tensor<128x128x28x28xf32>
  }
}

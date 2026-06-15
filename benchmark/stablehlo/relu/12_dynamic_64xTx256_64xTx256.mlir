module @f_12_dynamic_64xTx256_64xTx256 {
  func.func @f_12_dynamic_64xTx256_64xTx256(%input: tensor<64x?x256xf32>) -> tensor<64x?x256xf32> {
    %zero   = stablehlo.constant dense<0.0> : tensor<f32>
    %inf    = stablehlo.constant dense<3.4028235e+38> : tensor<f32>
    %result = stablehlo.clamp %zero, %input, %inf : (tensor<f32>, tensor<64x?x256xf32>, tensor<f32>) -> tensor<64x?x256xf32>
    return %result : tensor<64x?x256xf32>
  }
}

module @f_12_dynamic_64xTx256 {
  func.func @f_12_dynamic_64xTx256(%input: tensor<64x?x256xf32>) -> tensor<64x?x256xf32> {
    %result = stablehlo.logistic %input : tensor<64x?x256xf32>
    return %result : tensor<64x?x256xf32>
  }
}

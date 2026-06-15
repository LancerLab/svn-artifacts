module @f_12_dynamic_64xTx256_64xTx256_64xTx256 {
  func.func @f_12_dynamic_64xTx256_64xTx256_64xTx256(%input0: tensor<64x?x256xf32>, %input1: tensor<64x?x256xf32>) -> tensor<64x?x256xf32> {
    %result = stablehlo.add %input0, %input1 : tensor<64x?x256xf32>
    return %result : tensor<64x?x256xf32>
  }
}

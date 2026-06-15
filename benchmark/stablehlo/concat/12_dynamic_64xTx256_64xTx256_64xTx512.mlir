module @f_12_dynamic_64xTx256_64xTx256_64xTx512 {
  func.func @f_12_dynamic_64xTx256_64xTx256_64xTx512(%in0: tensor<64x?x256xf32>, %in1: tensor<64x?x256xf32>) -> tensor<64x?x512xf32> {
    %result = stablehlo.concatenate %in0, %in1, dim = 2 : (tensor<64x?x256xf32>, tensor<64x?x256xf32>) -> tensor<64x?x512xf32>
    return %result : tensor<64x?x512xf32>
  }
}

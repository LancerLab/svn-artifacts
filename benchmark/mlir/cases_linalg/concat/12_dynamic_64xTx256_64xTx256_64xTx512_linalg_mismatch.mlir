module {
  func.func @f_12_dynamic_64xTx256_64xTx256_64xTx512_mismatch(%in0: tensor<64x?x256xf32>, %in1: tensor<63x?x256xf32>) -> tensor<64x?x512xf32> {
    %r = tensor.concat dim(2) %in0, %in1 : (tensor<64x?x256xf32>, tensor<63x?x256xf32>) -> tensor<64x?x512xf32>
    return %r : tensor<64x?x512xf32>
  }
}

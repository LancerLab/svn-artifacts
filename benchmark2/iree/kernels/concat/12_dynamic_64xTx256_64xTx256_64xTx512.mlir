module {
  func.func @f_12_dynamic_64xTx256_64xTx256_64xTx512(%in0: tensor<64x?x256xf32>, %in1: tensor<64x?x256xf32>) -> tensor<64x?x512xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %in0_d1 = tensor.dim %in0, %c1 : tensor<64x?x256xf32>
    %in1_d1 = tensor.dim %in1, %c1 : tensor<64x?x256xf32>
    %out = tensor.empty(%in0_d1) : tensor<64x?x512xf32>
    %ins0 = tensor.insert_slice %in0 into %out[%c0, %c0, %c0][64, %in0_d1, 256][1, 1, 1] : tensor<64x?x256xf32> into tensor<64x?x512xf32>
    %coff256 = arith.constant 256 : index
    %ins1 = tensor.insert_slice %in1 into %ins0[%c0, %c0, %coff256][64, %in1_d1, 256][1, 1, 1] : tensor<64x?x256xf32> into tensor<64x?x512xf32>
    return %ins1 : tensor<64x?x512xf32>
  }
}

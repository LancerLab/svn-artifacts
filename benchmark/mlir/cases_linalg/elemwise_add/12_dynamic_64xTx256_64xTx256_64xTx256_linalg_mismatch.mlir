module {
  func.func @f_12_dynamic_64xTx256_64xTx256_64xTx256_mismatch(%a: tensor<64x?x256xf32>, %b: tensor<63x?x256xf32>) -> tensor<64x?x256xf32> {
    %c1 = arith.constant 1 : index
    %d1 = tensor.dim %a, %c1 : tensor<64x?x256xf32>
    %init = tensor.empty(%d1) : tensor<64x?x256xf32>
    %r = linalg.add ins(%a, %b : tensor<64x?x256xf32>, tensor<63x?x256xf32>)
                    outs(%init : tensor<64x?x256xf32>) -> tensor<64x?x256xf32>
    return %r : tensor<64x?x256xf32>
  }
}

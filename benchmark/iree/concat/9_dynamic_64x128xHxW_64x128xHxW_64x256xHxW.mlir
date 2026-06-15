module {
  func.func @f_9_dynamic_64x128xHxW_64x128xHxW_64x256xHxW(%in0: tensor<64x128x?x?xf32>, %in1: tensor<64x128x?x?xf32>) -> tensor<64x256x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %in0_d2 = tensor.dim %in0, %c2 : tensor<64x128x?x?xf32>
    %in0_d3 = tensor.dim %in0, %c3 : tensor<64x128x?x?xf32>
    %in1_d2 = tensor.dim %in1, %c2 : tensor<64x128x?x?xf32>
    %in1_d3 = tensor.dim %in1, %c3 : tensor<64x128x?x?xf32>
    %out = tensor.empty(%in0_d2, %in0_d3) : tensor<64x256x?x?xf32>
    %ins0 = tensor.insert_slice %in0 into %out[%c0, %c0, %c0, %c0][64, 128, %in0_d2, %in0_d3][1, 1, 1, 1] : tensor<64x128x?x?xf32> into tensor<64x256x?x?xf32>
    %coff128 = arith.constant 128 : index
    %ins1 = tensor.insert_slice %in1 into %ins0[%c0, %coff128, %c0, %c0][64, 128, %in1_d2, %in1_d3][1, 1, 1, 1] : tensor<64x128x?x?xf32> into tensor<64x256x?x?xf32>
    return %ins1 : tensor<64x256x?x?xf32>
  }
}

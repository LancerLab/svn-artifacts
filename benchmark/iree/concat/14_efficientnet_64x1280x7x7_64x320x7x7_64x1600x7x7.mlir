module {
  func.func @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7(%in0: tensor<64x1280x7x7xf32>, %in1: tensor<64x320x7x7xf32>) -> tensor<64x1600x7x7xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %out = tensor.empty() : tensor<64x1600x7x7xf32>
    %ins0 = tensor.insert_slice %in0 into %out[%c0, %c0, %c0, %c0][64, 1280, 7, 7][1, 1, 1, 1] : tensor<64x1280x7x7xf32> into tensor<64x1600x7x7xf32>
    %coff1280 = arith.constant 1280 : index
    %ins1 = tensor.insert_slice %in1 into %ins0[%c0, %coff1280, %c0, %c0][64, 320, 7, 7][1, 1, 1, 1] : tensor<64x320x7x7xf32> into tensor<64x1600x7x7xf32>
    return %ins1 : tensor<64x1600x7x7xf32>
  }
}

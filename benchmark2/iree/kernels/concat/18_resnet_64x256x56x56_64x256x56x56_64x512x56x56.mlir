module {
  func.func @f_18_resnet_64x256x56x56_64x256x56x56_64x512x56x56(%in0: tensor<64x256x56x56xf32>, %in1: tensor<64x256x56x56xf32>) -> tensor<64x512x56x56xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %out = tensor.empty() : tensor<64x512x56x56xf32>
    %ins0 = tensor.insert_slice %in0 into %out[%c0, %c0, %c0, %c0][64, 256, 56, 56][1, 1, 1, 1] : tensor<64x256x56x56xf32> into tensor<64x512x56x56xf32>
    %coff256 = arith.constant 256 : index
    %ins1 = tensor.insert_slice %in1 into %ins0[%c0, %coff256, %c0, %c0][64, 256, 56, 56][1, 1, 1, 1] : tensor<64x256x56x56xf32> into tensor<64x512x56x56xf32>
    return %ins1 : tensor<64x512x56x56xf32>
  }
}

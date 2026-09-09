module {
  func.func @f_21_vit_32x196x768_32x1x768_32x197x768(%in0: tensor<32x196x768xf32>, %in1: tensor<32x1x768xf32>) -> tensor<32x197x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %out = tensor.empty() : tensor<32x197x768xf32>
    %ins0 = tensor.insert_slice %in0 into %out[%c0, %c0, %c0][32, 196, 768][1, 1, 1] : tensor<32x196x768xf32> into tensor<32x197x768xf32>
    %coff196 = arith.constant 196 : index
    %ins1 = tensor.insert_slice %in1 into %ins0[%c0, %coff196, %c0][32, 1, 768][1, 1, 1] : tensor<32x1x768xf32> into tensor<32x197x768xf32>
    return %ins1 : tensor<32x197x768xf32>
  }
}

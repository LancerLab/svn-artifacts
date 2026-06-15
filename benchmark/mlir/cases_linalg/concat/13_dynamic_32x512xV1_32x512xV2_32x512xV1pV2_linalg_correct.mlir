module {
  func.func @f_13_dynamic_32x512xV1_32x512xV2_32x512xV1pV2_correct(%in0: tensor<32x512x?xf32>, %in1: tensor<32x512x?xf32>) -> tensor<32x512x?xf32> {
    %r = tensor.concat dim(2) %in0, %in1 : (tensor<32x512x?xf32>, tensor<32x512x?xf32>) -> tensor<32x512x?xf32>
    return %r : tensor<32x512x?xf32>
  }
}

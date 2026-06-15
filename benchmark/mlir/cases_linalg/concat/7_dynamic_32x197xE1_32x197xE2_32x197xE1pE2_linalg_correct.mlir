module {
  func.func @f_7_dynamic_32x197xE1_32x197xE2_32x197xE1pE2_correct(%in0: tensor<32x197x?xf32>, %in1: tensor<32x197x?xf32>) -> tensor<32x197x?xf32> {
    %r = tensor.concat dim(2) %in0, %in1 : (tensor<32x197x?xf32>, tensor<32x197x?xf32>) -> tensor<32x197x?xf32>
    return %r : tensor<32x197x?xf32>
  }
}

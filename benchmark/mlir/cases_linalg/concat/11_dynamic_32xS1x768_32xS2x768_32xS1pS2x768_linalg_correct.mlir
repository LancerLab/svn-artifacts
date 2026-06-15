module {
  func.func @f_11_dynamic_32xS1x768_32xS2x768_32xS1pS2x768_correct(%in0: tensor<32x?x768xf32>, %in1: tensor<32x?x768xf32>) -> tensor<32x?x768xf32> {
    %r = tensor.concat dim(1) %in0, %in1 : (tensor<32x?x768xf32>, tensor<32x?x768xf32>) -> tensor<32x?x768xf32>
    return %r : tensor<32x?x768xf32>
  }
}

module {
  func.func @f_8_dynamic_16x1024xD1_16x1024xD2_16x1024xD1pD2_correct(%in0: tensor<16x1024x?xf32>, %in1: tensor<16x1024x?xf32>) -> tensor<16x1024x?xf32> {
    %r = tensor.concat dim(2) %in0, %in1 : (tensor<16x1024x?xf32>, tensor<16x1024x?xf32>) -> tensor<16x1024x?xf32>
    return %r : tensor<16x1024x?xf32>
  }
}

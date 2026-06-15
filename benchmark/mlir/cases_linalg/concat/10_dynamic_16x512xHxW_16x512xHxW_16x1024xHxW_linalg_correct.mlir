module {
  func.func @f_10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW_correct(%in0: tensor<16x512x?x?xf32>, %in1: tensor<16x512x?x?xf32>) -> tensor<16x1024x?x?xf32> {
    %r = tensor.concat dim(1) %in0, %in1 : (tensor<16x512x?x?xf32>, tensor<16x512x?x?xf32>) -> tensor<16x1024x?x?xf32>
    return %r : tensor<16x1024x?x?xf32>
  }
}

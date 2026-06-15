module {
  func.func @f_1_bert_32x512x768_32x512x768_32x512x1536_correct(%in0: tensor<32x512x768xf32>, %in1: tensor<32x512x768xf32>) -> tensor<32x512x1536xf32> {
    %r = tensor.concat dim(2) %in0, %in1 : (tensor<32x512x768xf32>, tensor<32x512x768xf32>) -> tensor<32x512x1536xf32>
    return %r : tensor<32x512x1536xf32>
  }
}

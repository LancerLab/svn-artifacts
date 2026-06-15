module @f_1_bert_32x512x768_32x512x768_32x512x1536 {
  func.func @f_1_bert_32x512x768_32x512x768_32x512x1536(%in0: tensor<32x512x768xf32>, %in1: tensor<32x512x768xf32>) -> tensor<32x512x1536xf32> {
    %result = stablehlo.concatenate %in0, %in1, dim = 2 : (tensor<32x512x768xf32>, tensor<32x512x768xf32>) -> tensor<32x512x1536xf32>
    return %result : tensor<32x512x1536xf32>
  }
}

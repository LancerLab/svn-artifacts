module @f_1_bert_32x512x768_768x768_32x512x768 {
  func.func @f_1_bert_32x512x768_768x768_32x512x768(%input0: tensor<32x512x768xf32>, %input1: tensor<768x768xf32>) -> tensor<32x512x768xf32> {
    %result = stablehlo.dot_general %input0, %input1,
        batching_dims = [] x [],
        contracting_dims = [2] x [0] : (tensor<32x512x768xf32>, tensor<768x768xf32>) -> tensor<32x512x768xf32>
    return %result : tensor<32x512x768xf32>
  }
}

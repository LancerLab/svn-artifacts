module @f_2_bert_32x512x30522_30522x768_32x512x768 {
  func.func @f_2_bert_32x512x30522_30522x768_32x512x768(%input0: tensor<32x512x30522xf32>, %input1: tensor<30522x768xf32>) -> tensor<32x512x768xf32> {
    %result = stablehlo.dot_general %input0, %input1,
        batching_dims = [] x [],
        contracting_dims = [2] x [0] : (tensor<32x512x30522xf32>, tensor<30522x768xf32>) -> tensor<32x512x768xf32>
    return %result : tensor<32x512x768xf32>
  }
}

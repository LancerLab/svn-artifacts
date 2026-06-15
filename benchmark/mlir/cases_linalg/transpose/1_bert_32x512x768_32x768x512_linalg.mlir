module {
  func.func @f_1_bert_32x512x768_32x768x512_linalg(%input: tensor<32x512x768xf32>) -> tensor<32x768x512xf32> {
    %init = tensor.empty() : tensor<32x768x512xf32>
    %r = linalg.transpose ins(%input : tensor<32x512x768xf32>) outs(%init : tensor<32x768x512xf32>) permutation = [0, 2, 1]
    return %r : tensor<32x768x512xf32>
  }
}

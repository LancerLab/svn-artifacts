module {
  func.func @f_1_bert_32x512x768_32x512x12x64(%input: tensor<32x512x768xf32>) -> tensor<32x512x12x64xf32> {
    %out = tensor.expand_shape %input [[0], [1], [2, 3]] output_shape [32, 512, 12, 64] : tensor<32x512x768xf32> into tensor<32x512x12x64xf32>
    return %out : tensor<32x512x12x64xf32>
  }
}

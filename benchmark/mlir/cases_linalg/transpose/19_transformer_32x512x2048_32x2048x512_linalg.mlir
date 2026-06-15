module {
  func.func @f_19_transformer_32x512x2048_32x2048x512_linalg(%input: tensor<32x512x2048xf32>) -> tensor<32x2048x512xf32> {
    %init = tensor.empty() : tensor<32x2048x512xf32>
    %r = linalg.transpose ins(%input : tensor<32x512x2048xf32>) outs(%init : tensor<32x2048x512xf32>) permutation = [0, 2, 1]
    return %r : tensor<32x2048x512xf32>
  }
}

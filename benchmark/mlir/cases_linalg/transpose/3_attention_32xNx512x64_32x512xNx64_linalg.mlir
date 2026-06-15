module {
  func.func @f_3_attention_32xNx512x64_32x512xNx64_linalg(%input: tensor<32x?x512x64xf32>) -> tensor<32x512x?x64xf32> {
    %c1_2 = arith.constant 1 : index
    %d2 = tensor.dim %input, %c1_2 : tensor<32x?x512x64xf32>
    %init = tensor.empty(%d2) : tensor<32x512x?x64xf32>
    %r = linalg.transpose ins(%input : tensor<32x?x512x64xf32>) outs(%init : tensor<32x512x?x64xf32>) permutation = [0, 2, 1, 3]
    return %r : tensor<32x512x?x64xf32>
  }
}

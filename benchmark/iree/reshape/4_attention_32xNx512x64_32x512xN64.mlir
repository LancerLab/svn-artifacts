module {
  func.func @f_4_attention_32xNx512x64_32x512xN64(%input: tensor<32x?x512x64xf32>) -> tensor<32x512x?xf32> {
    %c1 = arith.constant 1 : index
    %N = tensor.dim %input, %c1 : tensor<32x?x512x64xf32>
    %init = tensor.empty(%N) : tensor<32x512x?x64xf32>
    %transposed = linalg.transpose ins(%input : tensor<32x?x512x64xf32>)
                                   outs(%init : tensor<32x512x?x64xf32>)
                                   permutation = [0, 2, 1, 3]
    %out = tensor.collapse_shape %transposed [[0], [1], [2, 3]] : tensor<32x512x?x64xf32> into tensor<32x512x?xf32>
    return %out : tensor<32x512x?xf32>
  }
}

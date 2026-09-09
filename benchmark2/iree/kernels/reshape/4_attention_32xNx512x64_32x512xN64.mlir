module {
  func.func @f_4_attention_32xNx512x64_32x512xN64(%input: tensor<32x?x512x64xf32>) -> tensor<32x512x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<32x?x512x64xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<32x?x512x64xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<32x?x512x64xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<32x?x512x64xf32>
    %flat = tensor.collapse_shape %input [[0, 1, 2, 3]] : tensor<32x?x512x64xf32> into tensor<?xf32>
    %flat_sz = tensor.dim %flat, %c0 : tensor<?xf32>
    %cstat = arith.constant 16384 : index
    %dyn_dim = arith.divui %flat_sz, %cstat : index
    %out = tensor.expand_shape %flat [[0, 1, 2]] output_shape [32, 512, %dyn_dim] : tensor<?xf32> into tensor<32x512x?xf32>
    return %out : tensor<32x512x?xf32>
  }
}

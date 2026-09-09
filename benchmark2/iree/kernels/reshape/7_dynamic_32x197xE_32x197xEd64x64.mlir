module {
  func.func @f_7_dynamic_32x197xE_32x197xEd64x64(%input: tensor<32x197x?xf32>) -> tensor<32x197x?x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<32x197x?xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<32x197x?xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<32x197x?xf32>
    %flat = tensor.collapse_shape %input [[0, 1, 2]] : tensor<32x197x?xf32> into tensor<?xf32>
    %flat_sz = tensor.dim %flat, %c0 : tensor<?xf32>
    %cstat = arith.constant 403456 : index
    %dyn_dim = arith.divui %flat_sz, %cstat : index
    %out = tensor.expand_shape %flat [[0, 1, 2, 3]] output_shape [32, 197, %dyn_dim, 64] : tensor<?xf32> into tensor<32x197x?x64xf32>
    return %out : tensor<32x197x?x64xf32>
  }
}

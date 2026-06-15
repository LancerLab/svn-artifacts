module {
  func.func @f_7_dynamic_32x197xE_32x197xEd64x64(%input: tensor<32x197x?xf32>) -> tensor<32x197x?x64xf32> {
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %E = tensor.dim %input, %c2 : tensor<32x197x?xf32>
    %E_div_64 = arith.divui %E, %c64 : index
    %out = tensor.expand_shape %input [[0], [1], [2, 3]] output_shape [32, 197, %E_div_64, 64] : tensor<32x197x?xf32> into tensor<32x197x?x64xf32>
    return %out : tensor<32x197x?x64xf32>
  }
}

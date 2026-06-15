module {
  func.func @f_8_dynamic_16x1024xD_16x1024xDd64x64(%input: tensor<16x1024x?xf32>) -> tensor<16x1024x?x64xf32> {
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %D = tensor.dim %input, %c2 : tensor<16x1024x?xf32>
    %D_div_64 = arith.divui %D, %c64 : index
    %out = tensor.expand_shape %input [[0], [1], [2, 3]] output_shape [16, 1024, %D_div_64, 64] : tensor<16x1024x?xf32> into tensor<16x1024x?x64xf32>
    return %out : tensor<16x1024x?x64xf32>
  }
}

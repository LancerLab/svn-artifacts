module {
  func.func @f_3_attention_BxSx768_BxSx12x64(%input: tensor<?x?x768xf32>) -> tensor<?x?x12x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<?x?x768xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<?x?x768xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<?x?x768xf32>
    %out = tensor.expand_shape %input [[0], [1], [2, 3]] output_shape [%input_d0, %input_d1, 12, 64] : tensor<?x?x768xf32> into tensor<?x?x12x64xf32>
    return %out : tensor<?x?x12x64xf32>
  }
}

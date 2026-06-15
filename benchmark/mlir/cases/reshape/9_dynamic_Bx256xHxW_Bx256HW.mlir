module {
  func.func @f_9_dynamic_Bx256xHxW_Bx256HW(%input: tensor<?x256x?x?xf32>) -> tensor<?x?xf32> {
    %out = tensor.collapse_shape %input [[0], [1, 2, 3]] : tensor<?x256x?x?xf32> into tensor<?x?xf32>
    return %out : tensor<?x?xf32>
  }
}

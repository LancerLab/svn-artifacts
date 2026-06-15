module {
  func.func @f_5_dynamic_Bx1280xHxW_Bx1280HW(%input: tensor<?x1280x?x?xf32>) -> tensor<?x?xf32> {
    %out = tensor.collapse_shape %input [[0], [1, 2, 3]] : tensor<?x1280x?x?xf32> into tensor<?x?xf32>
    return %out : tensor<?x?xf32>
  }
}

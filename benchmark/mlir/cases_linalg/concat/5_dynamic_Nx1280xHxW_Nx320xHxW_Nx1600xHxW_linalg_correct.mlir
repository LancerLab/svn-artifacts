module {
  func.func @f_5_dynamic_Nx1280xHxW_Nx320xHxW_Nx1600xHxW_correct(%in0: tensor<?x1280x?x?xf32>, %in1: tensor<?x320x?x?xf32>) -> tensor<?x1600x?x?xf32> {
    %r = tensor.concat dim(1) %in0, %in1 : (tensor<?x1280x?x?xf32>, tensor<?x320x?x?xf32>) -> tensor<?x1600x?x?xf32>
    return %r : tensor<?x1600x?x?xf32>
  }
}

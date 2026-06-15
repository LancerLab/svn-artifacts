module @f_5_dynamic_Nx1280xHxW_Nx320xHxW_Nx1600xHxW {
  func.func @f_5_dynamic_Nx1280xHxW_Nx320xHxW_Nx1600xHxW(%in0: tensor<?x1280x?x?xf32>, %in1: tensor<?x320x?x?xf32>) -> tensor<?x1600x?x?xf32> {
    %result = stablehlo.concatenate %in0, %in1, dim = 1 : (tensor<?x1280x?x?xf32>, tensor<?x320x?x?xf32>) -> tensor<?x1600x?x?xf32>
    return %result : tensor<?x1600x?x?xf32>
  }
}

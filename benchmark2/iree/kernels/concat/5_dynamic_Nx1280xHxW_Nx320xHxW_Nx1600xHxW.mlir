module {
  func.func @f_5_dynamic_Nx1280xHxW_Nx320xHxW_Nx1600xHxW(%in0: tensor<?x1280x?x?xf32>, %in1: tensor<?x320x?x?xf32>) -> tensor<?x1600x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %in0_d0 = tensor.dim %in0, %c0 : tensor<?x1280x?x?xf32>
    %in0_d2 = tensor.dim %in0, %c2 : tensor<?x1280x?x?xf32>
    %in0_d3 = tensor.dim %in0, %c3 : tensor<?x1280x?x?xf32>
    %in1_d0 = tensor.dim %in1, %c0 : tensor<?x320x?x?xf32>
    %in1_d2 = tensor.dim %in1, %c2 : tensor<?x320x?x?xf32>
    %in1_d3 = tensor.dim %in1, %c3 : tensor<?x320x?x?xf32>
    %out = tensor.empty(%in0_d0, %in0_d2, %in0_d3) : tensor<?x1600x?x?xf32>
    %ins0 = tensor.insert_slice %in0 into %out[%c0, %c0, %c0, %c0][%in0_d0, 1280, %in0_d2, %in0_d3][1, 1, 1, 1] : tensor<?x1280x?x?xf32> into tensor<?x1600x?x?xf32>
    %coff1280 = arith.constant 1280 : index
    %ins1 = tensor.insert_slice %in1 into %ins0[%c0, %coff1280, %c0, %c0][%in1_d0, 320, %in1_d2, %in1_d3][1, 1, 1, 1] : tensor<?x320x?x?xf32> into tensor<?x1600x?x?xf32>
    return %ins1 : tensor<?x1600x?x?xf32>
  }
}

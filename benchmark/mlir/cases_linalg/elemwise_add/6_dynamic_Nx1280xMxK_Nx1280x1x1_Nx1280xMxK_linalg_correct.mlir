module {
  func.func @f_6_dynamic_Nx1280xMxK_Nx1280x1x1_Nx1280xMxK_correct(%a: tensor<?x1280x?x?xf32>, %b: tensor<?x1280x1x1xf32>) -> tensor<?x1280x?x?xf32> {
    %c0 = arith.constant 0 : index
    %d0 = tensor.dim %a, %c0 : tensor<?x1280x?x?xf32>
    %c2 = arith.constant 2 : index
    %d2 = tensor.dim %a, %c2 : tensor<?x1280x?x?xf32>
    %c3 = arith.constant 3 : index
    %d3 = tensor.dim %a, %c3 : tensor<?x1280x?x?xf32>
    %init = tensor.empty(%d0, %d2, %d3) : tensor<?x1280x?x?xf32>
    %r = linalg.add ins(%a, %b : tensor<?x1280x?x?xf32>, tensor<?x1280x1x1xf32>)
                    outs(%init : tensor<?x1280x?x?xf32>) -> tensor<?x1280x?x?xf32>
    return %r : tensor<?x1280x?x?xf32>
  }
}

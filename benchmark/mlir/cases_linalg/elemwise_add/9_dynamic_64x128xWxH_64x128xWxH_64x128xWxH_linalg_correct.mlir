module {
  func.func @f_9_dynamic_64x128xWxH_64x128xWxH_64x128xWxH_correct(%a: tensor<64x128x?x?xf32>, %b: tensor<64x128x?x?xf32>) -> tensor<64x128x?x?xf32> {
    %c2 = arith.constant 2 : index
    %d2 = tensor.dim %a, %c2 : tensor<64x128x?x?xf32>
    %c3 = arith.constant 3 : index
    %d3 = tensor.dim %a, %c3 : tensor<64x128x?x?xf32>
    %init = tensor.empty(%d2, %d3) : tensor<64x128x?x?xf32>
    %r = linalg.add ins(%a, %b : tensor<64x128x?x?xf32>, tensor<64x128x?x?xf32>)
                    outs(%init : tensor<64x128x?x?xf32>) -> tensor<64x128x?x?xf32>
    return %r : tensor<64x128x?x?xf32>
  }
}

module {
  func.func @f_13_dynamic_32x512xV_32x512xV_32x512xV_correct(%a: tensor<32x512x?xf32>, %b: tensor<32x512x?xf32>) -> tensor<32x512x?xf32> {
    %c2 = arith.constant 2 : index
    %d2 = tensor.dim %a, %c2 : tensor<32x512x?xf32>
    %init = tensor.empty(%d2) : tensor<32x512x?xf32>
    %r = linalg.add ins(%a, %b : tensor<32x512x?xf32>, tensor<32x512x?xf32>)
                    outs(%init : tensor<32x512x?xf32>) -> tensor<32x512x?xf32>
    return %r : tensor<32x512x?xf32>
  }
}

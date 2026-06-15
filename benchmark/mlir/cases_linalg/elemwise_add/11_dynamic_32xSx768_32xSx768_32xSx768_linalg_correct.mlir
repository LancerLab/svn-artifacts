module {
  func.func @f_11_dynamic_32xSx768_32xSx768_32xSx768_correct(%a: tensor<32x?x768xf32>, %b: tensor<32x?x768xf32>) -> tensor<32x?x768xf32> {
    %c1 = arith.constant 1 : index
    %d1 = tensor.dim %a, %c1 : tensor<32x?x768xf32>
    %init = tensor.empty(%d1) : tensor<32x?x768xf32>
    %r = linalg.add ins(%a, %b : tensor<32x?x768xf32>, tensor<32x?x768xf32>)
                    outs(%init : tensor<32x?x768xf32>) -> tensor<32x?x768xf32>
    return %r : tensor<32x?x768xf32>
  }
}

module {
  func.func @f_8_dynamic_16x1024xD_16x1024xD_16x1024xD_correct(%a: tensor<16x1024x?xf32>, %b: tensor<16x1024x?xf32>) -> tensor<16x1024x?xf32> {
    %c2 = arith.constant 2 : index
    %d2 = tensor.dim %a, %c2 : tensor<16x1024x?xf32>
    %init = tensor.empty(%d2) : tensor<16x1024x?xf32>
    %r = linalg.add ins(%a, %b : tensor<16x1024x?xf32>, tensor<16x1024x?xf32>)
                    outs(%init : tensor<16x1024x?xf32>) -> tensor<16x1024x?xf32>
    return %r : tensor<16x1024x?xf32>
  }
}

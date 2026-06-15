module {
  func.func @f_10_dynamic_16x512xHxW_16x512xHxW_16x512xHxW_mismatch(%a: tensor<16x512x?x?xf32>, %b: tensor<15x512x?x?xf32>) -> tensor<16x512x?x?xf32> {
    %c2 = arith.constant 2 : index
    %d2 = tensor.dim %a, %c2 : tensor<16x512x?x?xf32>
    %c3 = arith.constant 3 : index
    %d3 = tensor.dim %a, %c3 : tensor<16x512x?x?xf32>
    %init = tensor.empty(%d2, %d3) : tensor<16x512x?x?xf32>
    %r = linalg.add ins(%a, %b : tensor<16x512x?x?xf32>, tensor<15x512x?x?xf32>)
                    outs(%init : tensor<16x512x?x?xf32>) -> tensor<16x512x?x?xf32>
    return %r : tensor<16x512x?x?xf32>
  }
}

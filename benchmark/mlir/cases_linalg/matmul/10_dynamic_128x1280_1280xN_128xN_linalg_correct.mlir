module {
  func.func @f_10_dynamic_128x1280_1280xN_128xN_correct(%a: tensor<128x1280xf32>, %b: tensor<1280x?xf32>) -> tensor<128x?xf32> {
    %cst = arith.constant 0.0 : f32
    %c1 = arith.constant 1 : index
    %n = tensor.dim %b, %c1 : tensor<1280x?xf32>
    %init = tensor.empty(%n) : tensor<128x?xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x?xf32>) -> tensor<128x?xf32>
    %r = linalg.matmul ins(%a, %b : tensor<128x1280xf32>, tensor<1280x?xf32>)
                        outs(%fill : tensor<128x?xf32>) -> tensor<128x?xf32>
    return %r : tensor<128x?xf32>
  }
}

module {
  func.func @f_14_efficientnet_64x1280_1280x1000_64x1000_correct(%a: tensor<64x1280xf32>, %b: tensor<1280x1000xf32>) -> tensor<64x1000xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<64x1000xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<64x1000xf32>) -> tensor<64x1000xf32>
    %r = linalg.matmul ins(%a, %b : tensor<64x1280xf32>, tensor<1280x1000xf32>)
                        outs(%fill : tensor<64x1000xf32>) -> tensor<64x1000xf32>
    return %r : tensor<64x1000xf32>
  }
}

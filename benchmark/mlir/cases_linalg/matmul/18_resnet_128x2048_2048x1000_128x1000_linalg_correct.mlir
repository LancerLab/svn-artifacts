module {
  func.func @f_18_resnet_128x2048_2048x1000_128x1000_correct(%a: tensor<128x2048xf32>, %b: tensor<2048x1000xf32>) -> tensor<128x1000xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<128x1000xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x1000xf32>) -> tensor<128x1000xf32>
    %r = linalg.matmul ins(%a, %b : tensor<128x2048xf32>, tensor<2048x1000xf32>)
                        outs(%fill : tensor<128x1000xf32>) -> tensor<128x1000xf32>
    return %r : tensor<128x1000xf32>
  }
}

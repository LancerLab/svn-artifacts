module {
  func.func @f_17_mobilenet_128x96x112x112_128x96x112x112_128x96x112x112_correct(%a: tensor<128x96x112x112xf32>, %b: tensor<128x96x112x112xf32>) -> tensor<128x96x112x112xf32> {
    %init = tensor.empty() : tensor<128x96x112x112xf32>
    %r = linalg.add ins(%a, %b : tensor<128x96x112x112xf32>, tensor<128x96x112x112xf32>)
                    outs(%init : tensor<128x96x112x112xf32>) -> tensor<128x96x112x112xf32>
    return %r : tensor<128x96x112x112xf32>
  }
}

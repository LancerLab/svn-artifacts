module {
  func.func @f_7_dynamic_128xCx112x112_128xCx112x112_128xCx112x112_mismatch(%a: tensor<128x?x112x112xf32>, %b: tensor<127x?x112x112xf32>) -> tensor<128x?x112x112xf32> {
    %c1 = arith.constant 1 : index
    %d1 = tensor.dim %a, %c1 : tensor<128x?x112x112xf32>
    %init = tensor.empty(%d1) : tensor<128x?x112x112xf32>
    %r = linalg.add ins(%a, %b : tensor<128x?x112x112xf32>, tensor<127x?x112x112xf32>)
                    outs(%init : tensor<128x?x112x112xf32>) -> tensor<128x?x112x112xf32>
    return %r : tensor<128x?x112x112xf32>
  }
}

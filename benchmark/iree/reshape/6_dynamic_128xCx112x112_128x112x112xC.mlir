module {
  func.func @f_6_dynamic_128xCx112x112_128x112x112xC(%input: tensor<128x?x112x112xf32>) -> tensor<128x112x112x?xf32> {
    %c1 = arith.constant 1 : index
    %C = tensor.dim %input, %c1 : tensor<128x?x112x112xf32>
    %init = tensor.empty(%C) : tensor<128x112x112x?xf32>
    %out = linalg.transpose ins(%input : tensor<128x?x112x112xf32>)
                            outs(%init : tensor<128x112x112x?xf32>)
                            permutation = [0, 2, 3, 1]
    return %out : tensor<128x112x112x?xf32>
  }
}

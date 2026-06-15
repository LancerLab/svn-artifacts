module {
  func.func @f_6_dynamic_128xCx112x112_128x112x112xC_linalg(%input: tensor<128x?x112x112xf32>) -> tensor<128x112x112x?xf32> {
    %c1_3 = arith.constant 1 : index
    %d3 = tensor.dim %input, %c1_3 : tensor<128x?x112x112xf32>
    %init = tensor.empty(%d3) : tensor<128x112x112x?xf32>
    %r = linalg.transpose ins(%input : tensor<128x?x112x112xf32>) outs(%init : tensor<128x112x112x?xf32>) permutation = [0, 2, 3, 1]
    return %r : tensor<128x112x112x?xf32>
  }
}

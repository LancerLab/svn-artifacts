module {
  func.func @f_6_dynamic_128xCx112x112_128x112x112xC(%input: tensor<128x?x112x112xf32>) -> tensor<128x112x112x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<128x?x112x112xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<128x?x112x112xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<128x?x112x112xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<128x?x112x112xf32>
    %out = tensor.empty(%input_d1) : tensor<128x112x112x?xf32>
    %result = linalg.transpose ins(%input : tensor<128x?x112x112xf32>) outs(%out : tensor<128x112x112x?xf32>) permutation = [0, 2, 3, 1]
    return %result : tensor<128x112x112x?xf32>
  }
}

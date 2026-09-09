module {
  func.func @f_2_cnn_128x128x28x28_128x28x28x128(%input: tensor<128x128x28x28xf32>) -> tensor<128x28x28x128xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<128x128x28x28xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<128x128x28x28xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<128x128x28x28xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<128x128x28x28xf32>
    %out = tensor.empty() : tensor<128x28x28x128xf32>
    %result = linalg.transpose ins(%input : tensor<128x128x28x28xf32>) outs(%out : tensor<128x28x28x128xf32>) permutation = [0, 2, 3, 1]
    return %result : tensor<128x28x28x128xf32>
  }
}

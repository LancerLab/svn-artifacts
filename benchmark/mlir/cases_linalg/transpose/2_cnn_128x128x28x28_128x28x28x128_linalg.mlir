module {
  func.func @f_2_cnn_128x128x28x28_128x28x28x128_linalg(%input: tensor<128x128x28x28xf32>) -> tensor<128x28x28x128xf32> {
    %init = tensor.empty() : tensor<128x28x28x128xf32>
    %r = linalg.transpose ins(%input : tensor<128x128x28x28xf32>) outs(%init : tensor<128x28x28x128xf32>) permutation = [0, 2, 3, 1]
    return %r : tensor<128x28x28x128xf32>
  }
}

module {
  func.func @f_17_mobilenet_128xx96x112x112_128x112x112x96_linalg(%input: tensor<128x96x112x112xf32>) -> tensor<128x112x112x96xf32> {
    %init = tensor.empty() : tensor<128x112x112x96xf32>
    %r = linalg.transpose ins(%input : tensor<128x96x112x112xf32>) outs(%init : tensor<128x112x112x96xf32>) permutation = [0, 2, 3, 1]
    return %r : tensor<128x112x112x96xf32>
  }
}

module {
  func.func @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_correct(%in0: tensor<128x128x28x28xf32>, %in1: tensor<128x256x28x28xf32>) -> tensor<128x384x28x28xf32> {
    %r = tensor.concat dim(1) %in0, %in1 : (tensor<128x128x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x384x28x28xf32>
    return %r : tensor<128x384x28x28xf32>
  }
}

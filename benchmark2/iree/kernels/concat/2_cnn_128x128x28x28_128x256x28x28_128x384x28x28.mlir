module {
  func.func @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28(%in0: tensor<128x128x28x28xf32>, %in1: tensor<128x256x28x28xf32>) -> tensor<128x384x28x28xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %out = tensor.empty() : tensor<128x384x28x28xf32>
    %ins0 = tensor.insert_slice %in0 into %out[%c0, %c0, %c0, %c0][128, 128, 28, 28][1, 1, 1, 1] : tensor<128x128x28x28xf32> into tensor<128x384x28x28xf32>
    %coff128 = arith.constant 128 : index
    %ins1 = tensor.insert_slice %in1 into %ins0[%c0, %coff128, %c0, %c0][128, 256, 28, 28][1, 1, 1, 1] : tensor<128x256x28x28xf32> into tensor<128x384x28x28xf32>
    return %ins1 : tensor<128x384x28x28xf32>
  }
}

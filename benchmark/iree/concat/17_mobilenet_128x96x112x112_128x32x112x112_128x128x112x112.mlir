module {
  func.func @f_17_mobilenet_128x96x112x112_128x32x112x112_128x128x112x112(%in0: tensor<128x96x112x112xf32>, %in1: tensor<128x32x112x112xf32>) -> tensor<128x128x112x112xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %out = tensor.empty() : tensor<128x128x112x112xf32>
    %ins0 = tensor.insert_slice %in0 into %out[%c0, %c0, %c0, %c0][128, 96, 112, 112][1, 1, 1, 1] : tensor<128x96x112x112xf32> into tensor<128x128x112x112xf32>
    %coff96 = arith.constant 96 : index
    %ins1 = tensor.insert_slice %in1 into %ins0[%c0, %coff96, %c0, %c0][128, 32, 112, 112][1, 1, 1, 1] : tensor<128x32x112x112xf32> into tensor<128x128x112x112xf32>
    return %ins1 : tensor<128x128x112x112xf32>
  }
}

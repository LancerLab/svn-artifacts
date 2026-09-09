module {
  func.func @f_6_dynamic_128xC1x112x112_128xC2x112x112_128xC1pC2x112x112(%in0: tensor<128x?x112x112xf32>, %in1: tensor<128x?x112x112xf32>) -> tensor<128x?x112x112xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %in0_d1 = tensor.dim %in0, %c1 : tensor<128x?x112x112xf32>
    %in1_d1 = tensor.dim %in1, %c1 : tensor<128x?x112x112xf32>
    %csum0 = arith.addi %c0, %in0_d1 : index
    %csum1 = arith.addi %csum0, %in1_d1 : index
    %out = tensor.empty(%csum1) : tensor<128x?x112x112xf32>
    %ins0 = tensor.insert_slice %in0 into %out[%c0, %c0, %c0, %c0][128, %in0_d1, 112, 112][1, 1, 1, 1] : tensor<128x?x112x112xf32> into tensor<128x?x112x112xf32>
    %cum0 = arith.addi %c0, %in0_d1 : index
    %ins1 = tensor.insert_slice %in1 into %ins0[%c0, %cum0, %c0, %c0][128, %in1_d1, 112, 112][1, 1, 1, 1] : tensor<128x?x112x112xf32> into tensor<128x?x112x112xf32>
    return %ins1 : tensor<128x?x112x112xf32>
  }
}

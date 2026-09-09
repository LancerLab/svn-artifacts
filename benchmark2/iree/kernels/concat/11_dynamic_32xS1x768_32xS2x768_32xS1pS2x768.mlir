module {
  func.func @f_11_dynamic_32xS1x768_32xS2x768_32xS1pS2x768(%in0: tensor<32x?x768xf32>, %in1: tensor<32x?x768xf32>) -> tensor<32x?x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %in0_d1 = tensor.dim %in0, %c1 : tensor<32x?x768xf32>
    %in1_d1 = tensor.dim %in1, %c1 : tensor<32x?x768xf32>
    %csum0 = arith.addi %c0, %in0_d1 : index
    %csum1 = arith.addi %csum0, %in1_d1 : index
    %out = tensor.empty(%csum1) : tensor<32x?x768xf32>
    %ins0 = tensor.insert_slice %in0 into %out[%c0, %c0, %c0][32, %in0_d1, 768][1, 1, 1] : tensor<32x?x768xf32> into tensor<32x?x768xf32>
    %cum0 = arith.addi %c0, %in0_d1 : index
    %ins1 = tensor.insert_slice %in1 into %ins0[%c0, %cum0, %c0][32, %in1_d1, 768][1, 1, 1] : tensor<32x?x768xf32> into tensor<32x?x768xf32>
    return %ins1 : tensor<32x?x768xf32>
  }
}

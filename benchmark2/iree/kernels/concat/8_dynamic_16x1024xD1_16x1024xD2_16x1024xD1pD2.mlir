module {
  func.func @f_8_dynamic_16x1024xD1_16x1024xD2_16x1024xD1pD2(%in0: tensor<16x1024x?xf32>, %in1: tensor<16x1024x?xf32>) -> tensor<16x1024x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %in0_d2 = tensor.dim %in0, %c2 : tensor<16x1024x?xf32>
    %in1_d2 = tensor.dim %in1, %c2 : tensor<16x1024x?xf32>
    %csum0 = arith.addi %c0, %in0_d2 : index
    %csum1 = arith.addi %csum0, %in1_d2 : index
    %out = tensor.empty(%csum1) : tensor<16x1024x?xf32>
    %ins0 = tensor.insert_slice %in0 into %out[%c0, %c0, %c0][16, 1024, %in0_d2][1, 1, 1] : tensor<16x1024x?xf32> into tensor<16x1024x?xf32>
    %cum0 = arith.addi %c0, %in0_d2 : index
    %ins1 = tensor.insert_slice %in1 into %ins0[%c0, %c0, %cum0][16, 1024, %in1_d2][1, 1, 1] : tensor<16x1024x?xf32> into tensor<16x1024x?xf32>
    return %ins1 : tensor<16x1024x?xf32>
  }
}

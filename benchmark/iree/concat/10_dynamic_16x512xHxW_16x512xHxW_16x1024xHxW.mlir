module {
  func.func @f_10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW(%in0: tensor<16x512x?x?xf32>, %in1: tensor<16x512x?x?xf32>) -> tensor<16x1024x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %in0_d2 = tensor.dim %in0, %c2 : tensor<16x512x?x?xf32>
    %in0_d3 = tensor.dim %in0, %c3 : tensor<16x512x?x?xf32>
    %in1_d2 = tensor.dim %in1, %c2 : tensor<16x512x?x?xf32>
    %in1_d3 = tensor.dim %in1, %c3 : tensor<16x512x?x?xf32>
    %out = tensor.empty(%in0_d2, %in0_d3) : tensor<16x1024x?x?xf32>
    %ins0 = tensor.insert_slice %in0 into %out[%c0, %c0, %c0, %c0][16, 512, %in0_d2, %in0_d3][1, 1, 1, 1] : tensor<16x512x?x?xf32> into tensor<16x1024x?x?xf32>
    %coff512 = arith.constant 512 : index
    %ins1 = tensor.insert_slice %in1 into %ins0[%c0, %coff512, %c0, %c0][16, 512, %in1_d2, %in1_d3][1, 1, 1, 1] : tensor<16x512x?x?xf32> into tensor<16x1024x?x?xf32>
    return %ins1 : tensor<16x1024x?x?xf32>
  }
}

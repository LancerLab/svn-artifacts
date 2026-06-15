module {
  func.func @f_3_attention_32xNx512x64_32xNx512x64_32xNx512x128(%in0: tensor<32x?x512x64xf32>, %in1: tensor<32x?x512x64xf32>) -> tensor<32x?x512x128xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %in0_d1 = tensor.dim %in0, %c1 : tensor<32x?x512x64xf32>
    %in1_d1 = tensor.dim %in1, %c1 : tensor<32x?x512x64xf32>
    %out = tensor.empty(%in0_d1) : tensor<32x?x512x128xf32>
    %ins0 = tensor.insert_slice %in0 into %out[%c0, %c0, %c0, %c0][32, %in0_d1, 512, 64][1, 1, 1, 1] : tensor<32x?x512x64xf32> into tensor<32x?x512x128xf32>
    %coff64 = arith.constant 64 : index
    %ins1 = tensor.insert_slice %in1 into %ins0[%c0, %c0, %c0, %coff64][32, %in1_d1, 512, 64][1, 1, 1, 1] : tensor<32x?x512x64xf32> into tensor<32x?x512x128xf32>
    return %ins1 : tensor<32x?x512x128xf32>
  }
}

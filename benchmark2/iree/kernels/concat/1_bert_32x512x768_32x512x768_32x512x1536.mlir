module {
  func.func @f_1_bert_32x512x768_32x512x768_32x512x1536(%in0: tensor<32x512x768xf32>, %in1: tensor<32x512x768xf32>) -> tensor<32x512x1536xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %out = tensor.empty() : tensor<32x512x1536xf32>
    %ins0 = tensor.insert_slice %in0 into %out[%c0, %c0, %c0][32, 512, 768][1, 1, 1] : tensor<32x512x768xf32> into tensor<32x512x1536xf32>
    %coff768 = arith.constant 768 : index
    %ins1 = tensor.insert_slice %in1 into %ins0[%c0, %c0, %coff768][32, 512, 768][1, 1, 1] : tensor<32x512x768xf32> into tensor<32x512x1536xf32>
    return %ins1 : tensor<32x512x1536xf32>
  }
}

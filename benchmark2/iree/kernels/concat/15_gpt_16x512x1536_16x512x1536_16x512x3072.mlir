module {
  func.func @f_15_gpt_16x512x1536_16x512x1536_16x512x3072(%in0: tensor<16x512x1536xf32>, %in1: tensor<16x512x1536xf32>) -> tensor<16x512x3072xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %out = tensor.empty() : tensor<16x512x3072xf32>
    %ins0 = tensor.insert_slice %in0 into %out[%c0, %c0, %c0][16, 512, 1536][1, 1, 1] : tensor<16x512x1536xf32> into tensor<16x512x3072xf32>
    %coff1536 = arith.constant 1536 : index
    %ins1 = tensor.insert_slice %in1 into %ins0[%c0, %c0, %coff1536][16, 512, 1536][1, 1, 1] : tensor<16x512x1536xf32> into tensor<16x512x3072xf32>
    return %ins1 : tensor<16x512x3072xf32>
  }
}

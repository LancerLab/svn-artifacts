module {
  func.func @f_3_attention_32xNx512x64_32xNx512x64_32xNx512x128_mismatch(%in0: tensor<32x?x512x64xf32>, %in1: tensor<31x?x512x64xf32>) -> tensor<32x?x512x128xf32> {
    %r = tensor.concat dim(3) %in0, %in1 : (tensor<32x?x512x64xf32>, tensor<31x?x512x64xf32>) -> tensor<32x?x512x128xf32>
    return %r : tensor<32x?x512x128xf32>
  }
}

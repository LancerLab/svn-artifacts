module @f_17_general_32x512x512_512x512_32x512x512 {
  func.func @f_17_general_32x512x512_512x512_32x512x512(%input0: tensor<32x512x512xf32>, %input1: tensor<512x512xf32>) -> tensor<32x512x512xf32> {
    %result = stablehlo.dot_general %input0, %input1,
        batching_dims = [] x [],
        contracting_dims = [2] x [0] : (tensor<32x512x512xf32>, tensor<512x512xf32>) -> tensor<32x512x512xf32>
    return %result : tensor<32x512x512xf32>
  }
}

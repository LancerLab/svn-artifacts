module @f_4_dynamic_32xNx512x64_32xNx64x512_32xNx512x512 {
  func.func @f_4_dynamic_32xNx512x64_32xNx64x512_32xNx512x512(%input0: tensor<32x?x512x64xf32>, %input1: tensor<32x?x64x512xf32>) -> tensor<32x?x512x512xf32> {
    %result = stablehlo.dot_general %input0, %input1,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [2] : (tensor<32x?x512x64xf32>, tensor<32x?x64x512xf32>) -> tensor<32x?x512x512xf32>
    return %result : tensor<32x?x512x512xf32>
  }
}

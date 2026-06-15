module @f_20_vit_32x197x768_768x3072_32x197x3072 {
  func.func @f_20_vit_32x197x768_768x3072_32x197x3072(%input0: tensor<32x197x768xf32>, %input1: tensor<768x3072xf32>) -> tensor<32x197x3072xf32> {
    %result = stablehlo.dot_general %input0, %input1,
        batching_dims = [] x [],
        contracting_dims = [2] x [0] : (tensor<32x197x768xf32>, tensor<768x3072xf32>) -> tensor<32x197x3072xf32>
    return %result : tensor<32x197x3072xf32>
  }
}

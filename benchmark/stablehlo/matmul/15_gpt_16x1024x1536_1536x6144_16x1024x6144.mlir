module @f_15_gpt_16x1024x1536_1536x6144_16x1024x6144 {
  func.func @f_15_gpt_16x1024x1536_1536x6144_16x1024x6144(%input0: tensor<16x1024x1536xf32>, %input1: tensor<1536x6144xf32>) -> tensor<16x1024x6144xf32> {
    %result = stablehlo.dot_general %input0, %input1,
        batching_dims = [] x [],
        contracting_dims = [2] x [0] : (tensor<16x1024x1536xf32>, tensor<1536x6144xf32>) -> tensor<16x1024x6144xf32>
    return %result : tensor<16x1024x6144xf32>
  }
}

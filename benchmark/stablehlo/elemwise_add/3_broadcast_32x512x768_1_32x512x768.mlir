module @f_3_broadcast_32x512x768_1_32x512x768 {
  func.func @f_3_broadcast_32x512x768_1_32x512x768(%input0: tensor<32x512x768xf32>, %input1: tensor<1xf32>) -> tensor<32x512x768xf32> {
    %b_bcast = stablehlo.broadcast_in_dim %input1, dims = [0] : (tensor<1xf32>) -> tensor<32x512x768xf32>
    %result = stablehlo.add %input0, %b_bcast : tensor<32x512x768xf32>
    return %result : tensor<32x512x768xf32>
  }
}

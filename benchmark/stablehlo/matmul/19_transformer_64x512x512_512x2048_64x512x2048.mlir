module @f_19_transformer_64x512x512_512x2048_64x512x2048 {
  func.func @f_19_transformer_64x512x512_512x2048_64x512x2048(%input0: tensor<64x512x512xf32>, %input1: tensor<512x2048xf32>) -> tensor<64x512x2048xf32> {
    %result = stablehlo.dot_general %input0, %input1,
        batching_dims = [] x [],
        contracting_dims = [2] x [0] : (tensor<64x512x512xf32>, tensor<512x2048xf32>) -> tensor<64x512x2048xf32>
    return %result : tensor<64x512x2048xf32>
  }
}

module @f_18_transformer_32x512_32000x512_32x512x512 {
  func.func @f_18_transformer_32x512_32000x512_32x512x512(%table: tensor<32000x512xf32>, %indices: tensor<32x512xi32>) -> tensor<32x512x512xf32> {
    %result = "stablehlo.gather"(%table, %indices)
        {dimension_numbers = #stablehlo.gather<
            offset_dims = [2],
            collapsed_slice_dims = [0],
            start_index_map = [0],
            index_vector_dim = 2>,
         slice_sizes = dense<[1, 512]> : tensor<2xi64>,
         indices_are_sorted = false}
        : (tensor<32000x512xf32>, tensor<32x512xi32>) -> tensor<32x512x512xf32>
    return %result : tensor<32x512x512xf32>
  }
}

module @f_16_general_64x256_20000x512_64x256x512 {
  func.func @f_16_general_64x256_20000x512_64x256x512(%table: tensor<20000x512xf32>, %indices: tensor<64x256xi32>) -> tensor<64x256x512xf32> {
    %result = "stablehlo.gather"(%table, %indices)
        {dimension_numbers = #stablehlo.gather<
            offset_dims = [2],
            collapsed_slice_dims = [0],
            start_index_map = [0],
            index_vector_dim = 2>,
         slice_sizes = dense<[1, 512]> : tensor<2xi64>,
         indices_are_sorted = false}
        : (tensor<20000x512xf32>, tensor<64x256xi32>) -> tensor<64x256x512xf32>
    return %result : tensor<64x256x512xf32>
  }
}

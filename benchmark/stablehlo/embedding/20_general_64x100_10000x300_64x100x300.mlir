module @f_20_general_64x100_10000x300_64x100x300 {
  func.func @f_20_general_64x100_10000x300_64x100x300(%table: tensor<10000x300xf32>, %indices: tensor<64x100xi32>) -> tensor<64x100x300xf32> {
    %result = "stablehlo.gather"(%table, %indices)
        {dimension_numbers = #stablehlo.gather<
            offset_dims = [2],
            collapsed_slice_dims = [0],
            start_index_map = [0],
            index_vector_dim = 2>,
         slice_sizes = dense<[1, 300]> : tensor<2xi64>,
         indices_are_sorted = false}
        : (tensor<10000x300xf32>, tensor<64x100xi32>) -> tensor<64x100x300xf32>
    return %result : tensor<64x100x300xf32>
  }
}

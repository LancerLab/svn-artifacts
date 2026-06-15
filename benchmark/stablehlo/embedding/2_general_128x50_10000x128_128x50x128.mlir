module @f_2_general_128x50_10000x128_128x50x128 {
  func.func @f_2_general_128x50_10000x128_128x50x128(%table: tensor<10000x128xf32>, %indices: tensor<128x50xi32>) -> tensor<128x50x128xf32> {
    %result = "stablehlo.gather"(%table, %indices)
        {dimension_numbers = #stablehlo.gather<
            offset_dims = [2],
            collapsed_slice_dims = [0],
            start_index_map = [0],
            index_vector_dim = 2>,
         slice_sizes = dense<[1, 128]> : tensor<2xi64>,
         indices_are_sorted = false}
        : (tensor<10000x128xf32>, tensor<128x50xi32>) -> tensor<128x50x128xf32>
    return %result : tensor<128x50x128xf32>
  }
}

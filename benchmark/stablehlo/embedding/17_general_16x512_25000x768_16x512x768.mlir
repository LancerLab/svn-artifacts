module @f_17_general_16x512_25000x768_16x512x768 {
  func.func @f_17_general_16x512_25000x768_16x512x768(%table: tensor<25000x768xf32>, %indices: tensor<16x512xi32>) -> tensor<16x512x768xf32> {
    %result = "stablehlo.gather"(%table, %indices)
        {dimension_numbers = #stablehlo.gather<
            offset_dims = [2],
            collapsed_slice_dims = [0],
            start_index_map = [0],
            index_vector_dim = 2>,
         slice_sizes = dense<[1, 768]> : tensor<2xi64>,
         indices_are_sorted = false}
        : (tensor<25000x768xf32>, tensor<16x512xi32>) -> tensor<16x512x768xf32>
    return %result : tensor<16x512x768xf32>
  }
}

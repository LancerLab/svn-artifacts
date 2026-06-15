module @f_6_dynamic_16xN_256x128_16xNx128 {
  func.func @f_6_dynamic_16xN_256x128_16xNx128(%table: tensor<256x128xf32>, %indices: tensor<16x?xi32>) -> tensor<16x?x128xf32> {
    %result = "stablehlo.gather"(%table, %indices)
        {dimension_numbers = #stablehlo.gather<
            offset_dims = [2],
            collapsed_slice_dims = [0],
            start_index_map = [0],
            index_vector_dim = 2>,
         slice_sizes = dense<[1, 128]> : tensor<2xi64>,
         indices_are_sorted = false}
        : (tensor<256x128xf32>, tensor<16x?xi32>) -> tensor<16x?x128xf32>
    return %result : tensor<16x?x128xf32>
  }
}

module @f_5_dynamic_32xN_50000x1024_32xNx1024 {
  func.func @f_5_dynamic_32xN_50000x1024_32xNx1024(%table: tensor<50000x1024xf32>, %indices: tensor<32x?xi32>) -> tensor<32x?x1024xf32> {
    %result = "stablehlo.gather"(%table, %indices)
        {dimension_numbers = #stablehlo.gather<
            offset_dims = [2],
            collapsed_slice_dims = [0],
            start_index_map = [0],
            index_vector_dim = 2>,
         slice_sizes = dense<[1, 1024]> : tensor<2xi64>,
         indices_are_sorted = false}
        : (tensor<50000x1024xf32>, tensor<32x?xi32>) -> tensor<32x?x1024xf32>
    return %result : tensor<32x?x1024xf32>
  }
}

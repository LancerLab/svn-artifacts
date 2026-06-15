module @f_9_dynamic_32xN_32000x768_32xNx768 {
  func.func @f_9_dynamic_32xN_32000x768_32xNx768(%table: tensor<32000x768xf32>, %indices: tensor<32x?xi32>) -> tensor<32x?x768xf32> {
    %result = "stablehlo.gather"(%table, %indices)
        {dimension_numbers = #stablehlo.gather<
            offset_dims = [2],
            collapsed_slice_dims = [0],
            start_index_map = [0],
            index_vector_dim = 2>,
         slice_sizes = dense<[1, 768]> : tensor<2xi64>,
         indices_are_sorted = false}
        : (tensor<32000x768xf32>, tensor<32x?xi32>) -> tensor<32x?x768xf32>
    return %result : tensor<32x?x768xf32>
  }
}

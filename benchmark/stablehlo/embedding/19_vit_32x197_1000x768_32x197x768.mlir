module @f_19_vit_32x197_1000x768_32x197x768 {
  func.func @f_19_vit_32x197_1000x768_32x197x768(%table: tensor<1000x768xf32>, %indices: tensor<32x197xi32>) -> tensor<32x197x768xf32> {
    %result = "stablehlo.gather"(%table, %indices)
        {dimension_numbers = #stablehlo.gather<
            offset_dims = [2],
            collapsed_slice_dims = [0],
            start_index_map = [0],
            index_vector_dim = 2>,
         slice_sizes = dense<[1, 768]> : tensor<2xi64>,
         indices_are_sorted = false}
        : (tensor<1000x768xf32>, tensor<32x197xi32>) -> tensor<32x197x768xf32>
    return %result : tensor<32x197x768xf32>
  }
}

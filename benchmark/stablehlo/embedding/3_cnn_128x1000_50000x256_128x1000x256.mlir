module @f_3_cnn_128x1000_50000x256_128x1000x256 {
  func.func @f_3_cnn_128x1000_50000x256_128x1000x256(%table: tensor<50000x256xf32>, %indices: tensor<128x1000xi32>) -> tensor<128x1000x256xf32> {
    %result = "stablehlo.gather"(%table, %indices)
        {dimension_numbers = #stablehlo.gather<
            offset_dims = [2],
            collapsed_slice_dims = [0],
            start_index_map = [0],
            index_vector_dim = 2>,
         slice_sizes = dense<[1, 256]> : tensor<2xi64>,
         indices_are_sorted = false}
        : (tensor<50000x256xf32>, tensor<128x1000xi32>) -> tensor<128x1000x256xf32>
    return %result : tensor<128x1000x256xf32>
  }
}

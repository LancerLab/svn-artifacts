module @f_14_gpt_16x1024_50257x1024_16x1024x1024 {
  func.func @f_14_gpt_16x1024_50257x1024_16x1024x1024(%table: tensor<50257x1024xf32>, %indices: tensor<16x1024xi32>) -> tensor<16x1024x1024xf32> {
    %result = "stablehlo.gather"(%table, %indices)
        {dimension_numbers = #stablehlo.gather<
            offset_dims = [2],
            collapsed_slice_dims = [0],
            start_index_map = [0],
            index_vector_dim = 2>,
         slice_sizes = dense<[1, 1024]> : tensor<2xi64>,
         indices_are_sorted = false}
        : (tensor<50257x1024xf32>, tensor<16x1024xi32>) -> tensor<16x1024x1024xf32>
    return %result : tensor<16x1024x1024xf32>
  }
}

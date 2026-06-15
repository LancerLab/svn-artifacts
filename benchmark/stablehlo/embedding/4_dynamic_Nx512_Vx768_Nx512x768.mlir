module @f_4_dynamic_Nx512_Vx768_Nx512x768 {
  func.func @f_4_dynamic_Nx512_Vx768_Nx512x768(%table: tensor<?x768xf32>, %indices: tensor<?x512xi32>) -> tensor<?x512x768xf32> {
    %result = "stablehlo.gather"(%table, %indices)
        {dimension_numbers = #stablehlo.gather<
            offset_dims = [2],
            collapsed_slice_dims = [0],
            start_index_map = [0],
            index_vector_dim = 2>,
         slice_sizes = dense<[1, 768]> : tensor<2xi64>,
         indices_are_sorted = false}
        : (tensor<?x768xf32>, tensor<?x512xi32>) -> tensor<?x512x768xf32>
    return %result : tensor<?x512x768xf32>
  }
}

module @f_15_bert_32x512_30522x768_32x512x768 {
  func.func @f_15_bert_32x512_30522x768_32x512x768(%table: tensor<30522x768xf32>, %indices: tensor<32x512xi32>) -> tensor<32x512x768xf32> {
    %result = "stablehlo.gather"(%table, %indices)
        {dimension_numbers = #stablehlo.gather<
            offset_dims = [2],
            collapsed_slice_dims = [0],
            start_index_map = [0],
            index_vector_dim = 2>,
         slice_sizes = dense<[1, 768]> : tensor<2xi64>,
         indices_are_sorted = false}
        : (tensor<30522x768xf32>, tensor<32x512xi32>) -> tensor<32x512x768xf32>
    return %result : tensor<32x512x768xf32>
  }
}

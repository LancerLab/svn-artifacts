module @f_19_transformer_32x512x2048_32x1048576 {
  func.func @f_19_transformer_32x512x2048_32x1048576(%input: memref<32x512x2048xf32>) -> memref<32x1048576xf32> {
    // reshape: no shape assertions required (element count preserved by type)
    %out = memref.collapse_shape %input [[0], [1, 2]] : memref<32x512x2048xf32> into memref<32x1048576xf32>
    return %out : memref<32x1048576xf32>
  }
}

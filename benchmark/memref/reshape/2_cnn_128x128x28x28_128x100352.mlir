module @f_2_cnn_128x128x28x28_128x100352 {
  func.func @f_2_cnn_128x128x28x28_128x100352(%input: memref<128x128x28x28xf32>) -> memref<128x100352xf32> {
    // reshape: no shape assertions required (element count preserved by type)
    %out = memref.collapse_shape %input [[0], [1, 2, 3]] : memref<128x128x28x28xf32> into memref<128x100352xf32>
    return %out : memref<128x100352xf32>
  }
}

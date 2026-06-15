module @f_18_resnet_64x512x7x7_64x25088 {
  func.func @f_18_resnet_64x512x7x7_64x25088(%input: memref<64x512x7x7xf32>) -> memref<64x25088xf32> {
    // reshape: no shape assertions required (element count preserved by type)
    %out = memref.collapse_shape %input [[0], [1, 2, 3]] : memref<64x512x7x7xf32> into memref<64x25088xf32>
    return %out : memref<64x25088xf32>
  }
}

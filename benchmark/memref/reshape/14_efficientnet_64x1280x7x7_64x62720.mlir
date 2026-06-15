module @f_14_efficientnet_64x1280x7x7_64x62720 {
  func.func @f_14_efficientnet_64x1280x7x7_64x62720(%input: memref<64x1280x7x7xf32>) -> memref<64x62720xf32> {
    // reshape: no shape assertions required (element count preserved by type)
    %out = memref.collapse_shape %input [[0], [1, 2, 3]] : memref<64x1280x7x7xf32> into memref<64x62720xf32>
    return %out : memref<64x62720xf32>
  }
}

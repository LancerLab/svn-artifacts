module @f_16_lstm_64x100x256_6400x256 {
  func.func @f_16_lstm_64x100x256_6400x256(%input: memref<64x100x256xf32>) -> memref<6400x256xf32> {
    // reshape: no shape assertions required (element count preserved by type)
    %out = memref.collapse_shape %input [[0, 1], [2]] : memref<64x100x256xf32> into memref<6400x256xf32>
    return %out : memref<6400x256xf32>
  }
}

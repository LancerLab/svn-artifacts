module {
  func.func @f_16_lstm_64x100x256_6400x256(%input: tensor<64x100x256xf32>) -> tensor<6400x256xf32> {
    %out = tensor.collapse_shape %input [[0, 1], [2]] : tensor<64x100x256xf32> into tensor<6400x256xf32>
    return %out : tensor<6400x256xf32>
  }
}

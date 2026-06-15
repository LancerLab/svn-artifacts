module {
  func.func @f_16_lstm_64x100x256_64x100x256_64x100x256_correct(%a: tensor<64x100x256xf32>, %b: tensor<64x100x256xf32>) -> tensor<64x100x256xf32> {
    %init = tensor.empty() : tensor<64x100x256xf32>
    %r = linalg.add ins(%a, %b : tensor<64x100x256xf32>, tensor<64x100x256xf32>)
                    outs(%init : tensor<64x100x256xf32>) -> tensor<64x100x256xf32>
    return %r : tensor<64x100x256xf32>
  }
}

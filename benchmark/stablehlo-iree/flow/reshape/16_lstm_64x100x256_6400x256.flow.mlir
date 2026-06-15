module @f_16_lstm_64x100x256_6400x256 {
  func.func @f_16_lstm_64x100x256_6400x256(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<64x100x256xf32>
    %1 = flow.tensor.reshape %0 : tensor<64x100x256xf32> -> tensor<6400x256xf32>
    %2 = hal.tensor.export %1 "output 0" : tensor<6400x256xf32> -> !hal.buffer_view
    return %2 : !hal.buffer_view
  }
}
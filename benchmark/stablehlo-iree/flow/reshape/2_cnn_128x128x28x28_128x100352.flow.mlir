module @f_2_cnn_128x128x28x28_128x100352 {
  func.func @f_2_cnn_128x128x28x28_128x100352(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<128x128x28x28xf32>
    %1 = flow.tensor.reshape %0 : tensor<128x128x28x28xf32> -> tensor<128x100352xf32>
    %2 = hal.tensor.export %1 "output 0" : tensor<128x100352xf32> -> !hal.buffer_view
    return %2 : !hal.buffer_view
  }
}
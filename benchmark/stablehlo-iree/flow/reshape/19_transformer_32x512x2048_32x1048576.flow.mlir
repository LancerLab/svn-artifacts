module @f_19_transformer_32x512x2048_32x1048576 {
  func.func @f_19_transformer_32x512x2048_32x1048576(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<32x512x2048xf32>
    %1 = flow.tensor.reshape %0 : tensor<32x512x2048xf32> -> tensor<32x1048576xf32>
    %2 = hal.tensor.export %1 "output 0" : tensor<32x1048576xf32> -> !hal.buffer_view
    return %2 : !hal.buffer_view
  }
}
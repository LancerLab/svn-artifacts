module @f_14_efficientnet_64x1280x7x7_64x62720 {
  func.func @f_14_efficientnet_64x1280x7x7_64x62720(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<64x1280x7x7xf32>
    %1 = flow.tensor.reshape %0 : tensor<64x1280x7x7xf32> -> tensor<64x62720xf32>
    %2 = hal.tensor.export %1 "output 0" : tensor<64x62720xf32> -> !hal.buffer_view
    return %2 : !hal.buffer_view
  }
}
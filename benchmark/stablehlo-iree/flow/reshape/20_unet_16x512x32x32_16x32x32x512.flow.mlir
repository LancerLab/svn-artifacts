module @f_20_unet_16x512x32x32_16x32x32x512 {
  func.func @f_20_unet_16x512x32x32_16x32x32x512(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<16x512x32x32xf32>
    %1 = flow.tensor.reshape %0 : tensor<16x512x32x32xf32> -> tensor<16x32x32x512xf32>
    %2 = hal.tensor.export %1 "output 0" : tensor<16x32x32x512xf32> -> !hal.buffer_view
    return %2 : !hal.buffer_view
  }
}
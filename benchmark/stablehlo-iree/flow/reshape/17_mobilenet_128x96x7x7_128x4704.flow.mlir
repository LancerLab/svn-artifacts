module @f_17_mobilenet_128x96x7x7_128x4704 {
  func.func @f_17_mobilenet_128x96x7x7_128x4704(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<128x96x7x7xf32>
    %1 = flow.tensor.reshape %0 : tensor<128x96x7x7xf32> -> tensor<128x4704xf32>
    %2 = hal.tensor.export %1 "output 0" : tensor<128x4704xf32> -> !hal.buffer_view
    return %2 : !hal.buffer_view
  }
}
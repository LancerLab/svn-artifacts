module @f_21_vit_32x3x224x224_32x196x768 {
  func.func @f_21_vit_32x3x224x224_32x196x768(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<32x3x224x224xf32>
    %1 = flow.tensor.reshape %0 : tensor<32x3x224x224xf32> -> tensor<32x196x768xf32>
    %2 = hal.tensor.export %1 "output 0" : tensor<32x196x768xf32> -> !hal.buffer_view
    return %2 : !hal.buffer_view
  }
}
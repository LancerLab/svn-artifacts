module @f_15_gpt_16x1024x1024_16x16x1024x64 {
  func.func @f_15_gpt_16x1024x1024_16x16x1024x64(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<16x1024x1024xf32>
    %1 = flow.tensor.reshape %0 : tensor<16x1024x1024xf32> -> tensor<16x16x1024x64xf32>
    %2 = hal.tensor.export %1 "output 0" : tensor<16x16x1024x64xf32> -> !hal.buffer_view
    return %2 : !hal.buffer_view
  }
}
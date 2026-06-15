module @f_1_bert_32x512x768_32x512x12x64 {
  func.func @f_1_bert_32x512x768_32x512x12x64(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<32x512x768xf32>
    %1 = flow.tensor.reshape %0 : tensor<32x512x768xf32> -> tensor<32x512x12x64xf32>
    %2 = hal.tensor.export %1 "output 0" : tensor<32x512x12x64xf32> -> !hal.buffer_view
    return %2 : !hal.buffer_view
  }
}
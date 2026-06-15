module @f_2_cnn_128x128x28x28_128x100352 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  func.func @f_2_cnn_128x128x28x28_128x100352(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c51380224 = arith.constant 51380224 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128 = arith.constant 128 : index
    %c28 = arith.constant 28 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c128, %c128, %c28, %c28]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<128x128x28x28xf32> in !stream.resource<external>{%c51380224}
    %1 = stream.tensor.export %0 : tensor<128x100352xf32> in !stream.resource<external>{%c51380224} -> !hal.buffer_view
    return %1 : !hal.buffer_view
  }
}
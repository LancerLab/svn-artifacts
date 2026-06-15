module @f_18_resnet_64x512x7x7_64x25088 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  func.func @f_18_resnet_64x512x7x7_64x25088(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c6422528 = arith.constant 6422528 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %c7 = arith.constant 7 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c64, %c512, %c7, %c7]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<64x512x7x7xf32> in !stream.resource<external>{%c6422528}
    %1 = stream.tensor.export %0 : tensor<64x25088xf32> in !stream.resource<external>{%c6422528} -> !hal.buffer_view
    return %1 : !hal.buffer_view
  }
}
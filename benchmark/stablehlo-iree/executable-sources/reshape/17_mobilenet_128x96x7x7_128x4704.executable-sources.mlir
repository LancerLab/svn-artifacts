module @f_17_mobilenet_128x96x7x7_128x4704 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  func.func @f_17_mobilenet_128x96x7x7_128x4704(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c2408448 = arith.constant 2408448 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128 = arith.constant 128 : index
    %c96 = arith.constant 96 : index
    %c7 = arith.constant 7 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c128, %c96, %c7, %c7]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<128x96x7x7xf32> in !stream.resource<external>{%c2408448}
    %1 = stream.tensor.export %0 : tensor<128x4704xf32> in !stream.resource<external>{%c2408448} -> !hal.buffer_view
    return %1 : !hal.buffer_view
  }
}
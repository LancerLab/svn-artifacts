module @f_19_transformer_32x512x2048_32x1048576 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  func.func @f_19_transformer_32x512x2048_32x1048576(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c134217728 = arith.constant 134217728 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c2048 = arith.constant 2048 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c32, %c512, %c2048]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<32x512x2048xf32> in !stream.resource<external>{%c134217728}
    %1 = stream.tensor.export %0 : tensor<32x1048576xf32> in !stream.resource<external>{%c134217728} -> !hal.buffer_view
    return %1 : !hal.buffer_view
  }
}
module @f_15_gpt_16x1024x1024_16x16x1024x64 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  func.func @f_15_gpt_16x1024x1024_16x16x1024x64(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c67108864 = arith.constant 67108864 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %c1024 = arith.constant 1024 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c16, %c1024, %c1024]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<16x1024x1024xf32> in !stream.resource<external>{%c67108864}
    %1 = stream.tensor.export %0 : tensor<16x16x1024x64xf32> in !stream.resource<external>{%c67108864} -> !hal.buffer_view
    return %1 : !hal.buffer_view
  }
}
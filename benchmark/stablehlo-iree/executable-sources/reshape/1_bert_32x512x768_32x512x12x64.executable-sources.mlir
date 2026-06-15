module @f_1_bert_32x512x768_32x512x12x64 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  func.func @f_1_bert_32x512x768_32x512x12x64(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c50331648 = arith.constant 50331648 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c768 = arith.constant 768 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c32, %c512, %c768]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<32x512x768xf32> in !stream.resource<external>{%c50331648}
    %1 = stream.tensor.export %0 : tensor<32x512x12x64xf32> in !stream.resource<external>{%c50331648} -> !hal.buffer_view
    return %1 : !hal.buffer_view
  }
}
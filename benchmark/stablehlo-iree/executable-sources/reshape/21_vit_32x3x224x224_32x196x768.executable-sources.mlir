module @f_21_vit_32x3x224x224_32x196x768 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  func.func @f_21_vit_32x3x224x224_32x196x768(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c19267584 = arith.constant 19267584 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32 = arith.constant 32 : index
    %c3 = arith.constant 3 : index
    %c224 = arith.constant 224 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c32, %c3, %c224, %c224]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<32x3x224x224xf32> in !stream.resource<external>{%c19267584}
    %1 = stream.tensor.export %0 : tensor<32x196x768xf32> in !stream.resource<external>{%c19267584} -> !hal.buffer_view
    return %1 : !hal.buffer_view
  }
}
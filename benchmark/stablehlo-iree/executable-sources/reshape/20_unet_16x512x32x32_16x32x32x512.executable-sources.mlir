module @f_20_unet_16x512x32x32_16x32x32x512 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  func.func @f_20_unet_16x512x32x32_16x32x32x512(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c33554432 = arith.constant 33554432 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index
    %c32 = arith.constant 32 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c16, %c512, %c32, %c32]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<16x512x32x32xf32> in !stream.resource<external>{%c33554432}
    %1 = stream.tensor.export %0 : tensor<16x32x32x512xf32> in !stream.resource<external>{%c33554432} -> !hal.buffer_view
    return %1 : !hal.buffer_view
  }
}
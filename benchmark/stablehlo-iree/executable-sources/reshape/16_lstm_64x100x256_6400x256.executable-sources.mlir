module @f_16_lstm_64x100x256_6400x256 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  func.func @f_16_lstm_64x100x256_6400x256(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c6553600 = arith.constant 6553600 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c100 = arith.constant 100 : index
    %c256 = arith.constant 256 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c64, %c100, %c256]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<64x100x256xf32> in !stream.resource<external>{%c6553600}
    %1 = stream.tensor.export %0 : tensor<6400x256xf32> in !stream.resource<external>{%c6553600} -> !hal.buffer_view
    return %1 : !hal.buffer_view
  }
}
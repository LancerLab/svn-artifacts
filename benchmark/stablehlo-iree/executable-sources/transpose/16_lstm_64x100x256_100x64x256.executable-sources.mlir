module @f_16_lstm_64x100x256_100x64x256 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_16_lstm_64x100x256_100x64x256_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_16_lstm_64x100x256_100x64x256_dispatch_0_generic_100x64x256_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_16_lstm_64x100x256_100x64x256_dispatch_0_generic_100x64x256_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x100x256xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<100x64x256xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 100, 256], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x100x256xf32>> -> tensor<64x100x256xf32>
          %3 = tensor.empty() : tensor<100x64x256xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<64x100x256xf32>) outs(%3 : tensor<100x64x256xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<100x64x256xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [100, 64, 256], strides = [1, 1, 1] : tensor<100x64x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<100x64x256xf32>>
          return
        }
      }
    }
  }
  func.func @f_16_lstm_64x100x256_100x64x256(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c6553600 = arith.constant 6553600 : index
    %c0 = arith.constant 0 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c100 = arith.constant 100 : index
    %c256 = arith.constant 256 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c64, %c100, %c256]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<64x100x256xf32> in !stream.resource<external>{%c6553600}
    %1 = stream.resource.alloc uninitialized : !stream.resource<external>{%c6553600}
    %2 = stream.cmd.execute with(%0 as %arg1: !stream.resource<external>{%c6553600}, %1 as %arg2: !stream.resource<external>{%c6553600}) {
      stream.cmd.dispatch @f_16_lstm_64x100x256_100x64x256_dispatch_0::@cuda_nvptx_fb::@f_16_lstm_64x100x256_100x64x256_dispatch_0_generic_100x64x256_f32 {
        ro %arg1[%c0 for %c6553600] : !stream.resource<external>{%c6553600},
        wo %arg2[%c0 for %c6553600] : !stream.resource<external>{%c6553600}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %3 = stream.timepoint.await %2 => %1 : !stream.resource<external>{%c6553600}
    %4 = stream.tensor.export %3 : tensor<100x64x256xf32> in !stream.resource<external>{%c6553600} -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
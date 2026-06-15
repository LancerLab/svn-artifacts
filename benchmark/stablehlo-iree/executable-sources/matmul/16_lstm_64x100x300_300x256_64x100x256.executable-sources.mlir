module @f_16_lstm_64x100x300_300x256_64x100x256 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_16_lstm_64x100x300_300x256_64x100x256_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_16_lstm_64x100x300_300x256_64x100x256_dispatch_0_matmul_6400x256x300_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_16_lstm_64x100x300_300x256_64x100x256_dispatch_0_matmul_6400x256x300_f32() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<6400x300xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<300x256xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<6400x256xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [6400, 300], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<6400x300xf32>> -> tensor<6400x300xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [300, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<300x256xf32>> -> tensor<300x256xf32>
          %5 = tensor.empty() : tensor<6400x256xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<6400x256xf32>) -> tensor<6400x256xf32>
          %7 = linalg.matmul ins(%3, %4 : tensor<6400x300xf32>, tensor<300x256xf32>) outs(%6 : tensor<6400x256xf32>) -> tensor<6400x256xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [6400, 256], strides = [1, 1] : tensor<6400x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<6400x256xf32>>
          return
        }
      }
    }
  }
  func.func @f_16_lstm_64x100x300_300x256_64x100x256(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c7680000 = arith.constant 7680000 : index
    %c307200 = arith.constant 307200 : index
    %c6553600 = arith.constant 6553600 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c100 = arith.constant 100 : index
    %c300 = arith.constant 300 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c64, %c100, %c300]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<64x100x300xf32> in !stream.resource<external>{%c7680000}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c300, %c256]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<300x256xf32> in !stream.resource<external>{%c307200}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c6553600}
    %3 = stream.cmd.execute with(%0 as %arg2: !stream.resource<external>{%c7680000}, %1 as %arg3: !stream.resource<external>{%c307200}, %2 as %arg4: !stream.resource<external>{%c6553600}) {
      stream.cmd.dispatch @f_16_lstm_64x100x300_300x256_64x100x256_dispatch_0::@cuda_nvptx_fb::@f_16_lstm_64x100x300_300x256_64x100x256_dispatch_0_matmul_6400x256x300_f32 {
        ro %arg2[%c0 for %c7680000] : !stream.resource<external>{%c7680000},
        ro %arg3[%c0 for %c307200] : !stream.resource<external>{%c307200},
        wo %arg4[%c0 for %c6553600] : !stream.resource<external>{%c6553600}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]}
    } => !stream.timepoint
    %4 = stream.timepoint.await %3 => %2 : !stream.resource<external>{%c6553600}
    %5 = stream.tensor.export %4 : tensor<64x100x256xf32> in !stream.resource<external>{%c6553600} -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
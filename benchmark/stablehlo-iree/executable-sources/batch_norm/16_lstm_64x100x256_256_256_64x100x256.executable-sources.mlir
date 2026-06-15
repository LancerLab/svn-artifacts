module @f_16_lstm_64x100x256_256_256_64x100x256 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_16_lstm_64x100x256_256_256_64x100x256_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_16_lstm_64x100x256_256_256_64x100x256_dispatch_0_generic_64x100x256_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer, ReadOnly>, <4, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_16_lstm_64x100x256_256_256_64x100x256_dispatch_0_generic_64x100x256_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x100x256xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256xf32>>
          %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256xf32>>
          %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x100x256xf32>>
          %5 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 100, 256], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x100x256xf32>> -> tensor<64x100x256xf32>
          %6 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [256], strides = [1] : !flow.dispatch.tensor<readonly:tensor<256xf32>> -> tensor<256xf32>
          %7 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [256], strides = [1] : !flow.dispatch.tensor<readonly:tensor<256xf32>> -> tensor<256xf32>
          %8 = flow.dispatch.tensor.load %3, offsets = [0], sizes = [256], strides = [1] : !flow.dispatch.tensor<readonly:tensor<256xf32>> -> tensor<256xf32>
          %9 = tensor.empty() : tensor<64x100x256xf32>
          %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %6, %7, %8 : tensor<64x100x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%9 : tensor<64x100x256xf32>) {
          ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
            %11 = arith.mulf %in, %in_0 : f32
            %12 = arith.divf %11, %in_1 : f32
            %13 = arith.addf %12, %in_2 : f32
            linalg.yield %13 : f32
          } -> tensor<64x100x256xf32>
          flow.dispatch.tensor.store %10, %4, offsets = [0, 0, 0], sizes = [64, 100, 256], strides = [1, 1, 1] : tensor<64x100x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x100x256xf32>>
          return
        }
      }
    }
  }
  func.func @f_16_lstm_64x100x256_256_256_64x100x256(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c1024 = arith.constant 1024 : index
    %c1065353258_i32 = arith.constant 1065353258 : i32
    %c6553600 = arith.constant 6553600 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c100 = arith.constant 100 : index
    %c64 = arith.constant 64 : index
    %c1_i32 = arith.constant 1 : i32
    %c553648160_i32 = arith.constant 553648160 : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c64, %c100, %c256]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<64x100x256xf32> in !stream.resource<external>{%c6553600}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c256]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<256xf32> in !stream.resource<external>{%c1024}
    hal.buffer_view.assert<%arg2 : !hal.buffer_view> message("input 2") shape([%c256]) type(%c553648160_i32) encoding(%c1_i32)
    %2 = stream.tensor.import %arg2 : !hal.buffer_view -> tensor<256xf32> in !stream.resource<external>{%c1024}
    %3 = stream.resource.alloc uninitialized : !stream.resource<external>{%c6553600}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c1024} => !stream.timepoint
    %4 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg3: !stream.resource<external>{%c6553600}, %1 as %arg4: !stream.resource<external>{%c1024}, %2 as %arg5: !stream.resource<external>{%c1024}, %3 as %arg6: !stream.resource<external>{%c6553600}, %result as %arg7: !stream.resource<transient>{%c1024}) {
      stream.cmd.fill %c1065353258_i32, %arg7[%c0 for %c1024] : i32 -> !stream.resource<transient>{%c1024}
      stream.cmd.dispatch @f_16_lstm_64x100x256_256_256_64x100x256_dispatch_0::@cuda_nvptx_fb::@f_16_lstm_64x100x256_256_256_64x100x256_dispatch_0_generic_64x100x256_f32 {
        ro %arg3[%c0 for %c6553600] : !stream.resource<external>{%c6553600},
        ro %arg4[%c0 for %c1024] : !stream.resource<external>{%c1024},
        ro %arg7[%c0 for %c1024] : !stream.resource<transient>{%c1024},
        ro %arg5[%c0 for %c1024] : !stream.resource<external>{%c1024},
        wo %arg6[%c0 for %c6553600] : !stream.resource<external>{%c6553600}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>, #hal.interface.binding<0, 4>]}
    } => !stream.timepoint
    %5 = stream.resource.dealloca await(%4) => %result : !stream.resource<transient>{%c1024} => !stream.timepoint
    %6 = stream.timepoint.await %5 => %3 : !stream.resource<external>{%c6553600}
    %7 = stream.tensor.export %6 : tensor<64x100x256xf32> in !stream.resource<external>{%c6553600} -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
}
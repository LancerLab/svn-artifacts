module @f_12_dynamic_64xTx256_256_256_64xTx256 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_12_dynamic_64xTx256_256_256_64xTx256_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_12_dynamic_64xTx256_256_256_64xTx256_dispatch_0_generic_64xDx256_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 1, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer, ReadOnly>, <4, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_12_dynamic_64xTx256_256_256_64xTx256_dispatch_0_generic_64xDx256_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = arith.index_castui %0 : i32 to index
          %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256xf32>>
          %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256xf32>>
          %4 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256xf32>>
          %5 = flow.dispatch.workload.ordinal %1, 0 : index
          %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x?x256xf32>>{%5}
          %7 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x?x256xf32>>{%5}
          %8 = flow.dispatch.tensor.load %6, offsets = [0, 0, 0], sizes = [64, %5, 256], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x?x256xf32>>{%5} -> tensor<64x?x256xf32>
          %9 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [256], strides = [1] : !flow.dispatch.tensor<readonly:tensor<256xf32>> -> tensor<256xf32>
          %10 = flow.dispatch.tensor.load %3, offsets = [0], sizes = [256], strides = [1] : !flow.dispatch.tensor<readonly:tensor<256xf32>> -> tensor<256xf32>
          %11 = flow.dispatch.tensor.load %4, offsets = [0], sizes = [256], strides = [1] : !flow.dispatch.tensor<readonly:tensor<256xf32>> -> tensor<256xf32>
          %12 = tensor.empty(%5) : tensor<64x?x256xf32>
          %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %9, %10, %11 : tensor<64x?x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%12 : tensor<64x?x256xf32>) {
          ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
            %14 = arith.mulf %in, %in_0 : f32
            %15 = arith.divf %14, %in_1 : f32
            %16 = arith.addf %15, %in_2 : f32
            linalg.yield %16 : f32
          } -> tensor<64x?x256xf32>
          flow.dispatch.tensor.store %13, %7, offsets = [0, 0, 0], sizes = [64, %5, 256], strides = [1, 1, 1] : tensor<64x?x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x?x256xf32>>{%5}
          return
        }
      }
    }
  }
  func.func @f_12_dynamic_64xTx256_256_256_64xTx256(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c1024 = arith.constant 1024 : index
    %c1065353258_i32 = arith.constant 1065353258 : i32
    %c65536 = arith.constant 65536 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %c1_i32 = arith.constant 1 : i32
    %c553648160_i32 = arith.constant 553648160 : i32
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c64, %0, %c256]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = arith.muli %0, %c65536 : index
    %2 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<64x?x256xf32>{%0} in !stream.resource<external>{%1}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c256]) type(%c553648160_i32) encoding(%c1_i32)
    %3 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<256xf32> in !stream.resource<external>{%c1024}
    hal.buffer_view.assert<%arg2 : !hal.buffer_view> message("input 2") shape([%c256]) type(%c553648160_i32) encoding(%c1_i32)
    %4 = stream.tensor.import %arg2 : !hal.buffer_view -> tensor<256xf32> in !stream.resource<external>{%c1024}
    %5 = stream.resource.alloc uninitialized : !stream.resource<external>{%1}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c1024} => !stream.timepoint
    %6 = arith.index_castui %0 : index to i32
    %7 = stream.cmd.execute await(%result_timepoint) => with(%2 as %arg3: !stream.resource<external>{%1}, %3 as %arg4: !stream.resource<external>{%c1024}, %4 as %arg5: !stream.resource<external>{%c1024}, %5 as %arg6: !stream.resource<external>{%1}, %result as %arg7: !stream.resource<transient>{%c1024}) {
      stream.cmd.fill %c1065353258_i32, %arg7[%c0 for %c1024] : i32 -> !stream.resource<transient>{%c1024}
      stream.cmd.dispatch @f_12_dynamic_64xTx256_256_256_64xTx256_dispatch_0::@cuda_nvptx_fb::@f_12_dynamic_64xTx256_256_256_64xTx256_dispatch_0_generic_64xDx256_f32[%0](%6 : i32) {
        ro %arg3[%c0 for %1] : !stream.resource<external>{%1},
        ro %arg4[%c0 for %c1024] : !stream.resource<external>{%c1024},
        ro %arg7[%c0 for %c1024] : !stream.resource<transient>{%c1024},
        ro %arg5[%c0 for %c1024] : !stream.resource<external>{%c1024},
        wo %arg6[%c0 for %1] : !stream.resource<external>{%1}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>, #hal.interface.binding<0, 4>]}
    } => !stream.timepoint
    %8 = stream.resource.dealloca await(%7) => %result : !stream.resource<transient>{%c1024} => !stream.timepoint
    %9 = stream.timepoint.await %8 => %5 : !stream.resource<external>{%1}
    %10 = stream.tensor.export %9 : tensor<64x?x256xf32>{%0} in !stream.resource<external>{%1} -> !hal.buffer_view
    return %10 : !hal.buffer_view
  }
}
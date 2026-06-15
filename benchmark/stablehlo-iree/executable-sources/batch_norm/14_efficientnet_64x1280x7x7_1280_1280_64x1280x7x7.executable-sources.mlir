module @f_14_efficientnet_64x1280x7x7_1280_1280_64x1280x7x7 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_14_efficientnet_64x1280x7x7_1280_1280_64x1280x7x7_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_14_efficientnet_64x1280x7x7_1280_1280_64x1280x7x7_dispatch_0_generic_64x1280x7x7_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer, ReadOnly>, <4, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_14_efficientnet_64x1280x7x7_1280_1280_64x1280x7x7_dispatch_0_generic_64x1280x7x7_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x1280x7x7xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xf32>>
          %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xf32>>
          %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x1280x7x7xf32>>
          %5 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [64, 1280, 7, 7], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x1280x7x7xf32>> -> tensor<64x1280x7x7xf32>
          %6 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf32>> -> tensor<1280xf32>
          %7 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf32>> -> tensor<1280xf32>
          %8 = flow.dispatch.tensor.load %3, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf32>> -> tensor<1280xf32>
          %9 = tensor.empty() : tensor<64x1280x7x7xf32>
          %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5, %6, %7, %8 : tensor<64x1280x7x7xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>) outs(%9 : tensor<64x1280x7x7xf32>) {
          ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
            %11 = arith.mulf %in, %in_0 : f32
            %12 = arith.divf %11, %in_1 : f32
            %13 = arith.addf %12, %in_2 : f32
            linalg.yield %13 : f32
          } -> tensor<64x1280x7x7xf32>
          flow.dispatch.tensor.store %10, %4, offsets = [0, 0, 0, 0], sizes = [64, 1280, 7, 7], strides = [1, 1, 1, 1] : tensor<64x1280x7x7xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x1280x7x7xf32>>
          return
        }
      }
    }
  }
  func.func @f_14_efficientnet_64x1280x7x7_1280_1280_64x1280x7x7(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c5120 = arith.constant 5120 : index
    %c1065353258_i32 = arith.constant 1065353258 : i32
    %c16056320 = arith.constant 16056320 : index
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c1280 = arith.constant 1280 : index
    %c64 = arith.constant 64 : index
    %c1_i32 = arith.constant 1 : i32
    %c553648160_i32 = arith.constant 553648160 : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c64, %c1280, %c7, %c7]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<64x1280x7x7xf32> in !stream.resource<external>{%c16056320}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c1280]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<1280xf32> in !stream.resource<external>{%c5120}
    hal.buffer_view.assert<%arg2 : !hal.buffer_view> message("input 2") shape([%c1280]) type(%c553648160_i32) encoding(%c1_i32)
    %2 = stream.tensor.import %arg2 : !hal.buffer_view -> tensor<1280xf32> in !stream.resource<external>{%c5120}
    %3 = stream.resource.alloc uninitialized : !stream.resource<external>{%c16056320}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c5120} => !stream.timepoint
    %4 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg3: !stream.resource<external>{%c16056320}, %1 as %arg4: !stream.resource<external>{%c5120}, %2 as %arg5: !stream.resource<external>{%c5120}, %3 as %arg6: !stream.resource<external>{%c16056320}, %result as %arg7: !stream.resource<transient>{%c5120}) {
      stream.cmd.fill %c1065353258_i32, %arg7[%c0 for %c5120] : i32 -> !stream.resource<transient>{%c5120}
      stream.cmd.dispatch @f_14_efficientnet_64x1280x7x7_1280_1280_64x1280x7x7_dispatch_0::@cuda_nvptx_fb::@f_14_efficientnet_64x1280x7x7_1280_1280_64x1280x7x7_dispatch_0_generic_64x1280x7x7_f32 {
        ro %arg3[%c0 for %c16056320] : !stream.resource<external>{%c16056320},
        ro %arg4[%c0 for %c5120] : !stream.resource<external>{%c5120},
        ro %arg7[%c0 for %c5120] : !stream.resource<transient>{%c5120},
        ro %arg5[%c0 for %c5120] : !stream.resource<external>{%c5120},
        wo %arg6[%c0 for %c16056320] : !stream.resource<external>{%c16056320}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>, #hal.interface.binding<0, 4>]}
    } => !stream.timepoint
    %5 = stream.resource.dealloca await(%4) => %result : !stream.resource<transient>{%c5120} => !stream.timepoint
    %6 = stream.timepoint.await %5 => %3 : !stream.resource<external>{%c16056320}
    %7 = stream.tensor.export %6 : tensor<64x1280x7x7xf32> in !stream.resource<external>{%c16056320} -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
}
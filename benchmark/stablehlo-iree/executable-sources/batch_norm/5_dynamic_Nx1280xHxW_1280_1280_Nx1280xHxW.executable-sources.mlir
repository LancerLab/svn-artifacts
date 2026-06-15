module @f_5_dynamic_Nx1280xHxW_1280_1280_Nx1280xHxW attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_5_dynamic_Nx1280xHxW_1280_1280_Nx1280xHxW_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_5_dynamic_Nx1280xHxW_1280_1280_Nx1280xHxW_dispatch_0_generic_Dx1280xDxD_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 3, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer, ReadOnly>, <4, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_5_dynamic_Nx1280xHxW_1280_1280_Nx1280xHxW_dispatch_0_generic_Dx1280xDxD_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = hal.interface.constant.load[2] : i32
          %3 = arith.index_castui %0 : i32 to index
          %4 = arith.index_castui %1 : i32 to index
          %5 = arith.index_castui %2 : i32 to index
          %6 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xf32>>
          %7 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xf32>>
          %8 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280xf32>>
          %9 = flow.dispatch.workload.ordinal %3, 0 : index
          %10 = flow.dispatch.workload.ordinal %4, 1 : index
          %11 = flow.dispatch.workload.ordinal %5, 2 : index
          %12 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x1280x?x?xf32>>{%9, %10, %11}
          %13 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x1280x?x?xf32>>{%9, %10, %11}
          %14 = flow.dispatch.tensor.load %12, offsets = [0, 0, 0, 0], sizes = [%9, 1280, %10, %11], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x1280x?x?xf32>>{%9, %10, %11} -> tensor<?x1280x?x?xf32>
          %15 = flow.dispatch.tensor.load %6, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf32>> -> tensor<1280xf32>
          %16 = flow.dispatch.tensor.load %7, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf32>> -> tensor<1280xf32>
          %17 = flow.dispatch.tensor.load %8, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf32>> -> tensor<1280xf32>
          %18 = tensor.empty(%9, %10, %11) : tensor<?x1280x?x?xf32>
          %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %15, %16, %17 : tensor<?x1280x?x?xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>) outs(%18 : tensor<?x1280x?x?xf32>) {
          ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
            %20 = arith.mulf %in, %in_0 : f32
            %21 = arith.divf %20, %in_1 : f32
            %22 = arith.addf %21, %in_2 : f32
            linalg.yield %22 : f32
          } -> tensor<?x1280x?x?xf32>
          flow.dispatch.tensor.store %19, %13, offsets = [0, 0, 0, 0], sizes = [%9, 1280, %10, %11], strides = [1, 1, 1, 1] : tensor<?x1280x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x1280x?x?xf32>>{%9, %10, %11}
          return
        }
      }
    }
  }
  func.func @f_5_dynamic_Nx1280xHxW_1280_1280_Nx1280xHxW(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c5120 = arith.constant 5120 : index
    %c1065353258_i32 = arith.constant 1065353258 : i32
    %c0 = arith.constant 0 : index
    %c1280 = arith.constant 1280 : index
    %c1_i32 = arith.constant 1 : i32
    %c553648160_i32 = arith.constant 553648160 : i32
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[2] : index
    %2 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[3] : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%0, %c1280, %1, %2]) type(%c553648160_i32) encoding(%c1_i32)
    %3 = arith.muli %0, %c5120 : index
    %4 = arith.muli %3, %1 : index
    %5 = arith.muli %4, %2 : index
    %6 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<?x1280x?x?xf32>{%0, %1, %2} in !stream.resource<external>{%5}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c1280]) type(%c553648160_i32) encoding(%c1_i32)
    %7 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<1280xf32> in !stream.resource<external>{%c5120}
    hal.buffer_view.assert<%arg2 : !hal.buffer_view> message("input 2") shape([%c1280]) type(%c553648160_i32) encoding(%c1_i32)
    %8 = stream.tensor.import %arg2 : !hal.buffer_view -> tensor<1280xf32> in !stream.resource<external>{%c5120}
    %9 = stream.resource.alloc uninitialized : !stream.resource<external>{%5}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c5120} => !stream.timepoint
    %10 = arith.index_castui %0 : index to i32
    %11 = arith.index_castui %1 : index to i32
    %12 = arith.index_castui %2 : index to i32
    %13 = stream.cmd.execute await(%result_timepoint) => with(%6 as %arg3: !stream.resource<external>{%5}, %7 as %arg4: !stream.resource<external>{%c5120}, %8 as %arg5: !stream.resource<external>{%c5120}, %9 as %arg6: !stream.resource<external>{%5}, %result as %arg7: !stream.resource<transient>{%c5120}) {
      stream.cmd.fill %c1065353258_i32, %arg7[%c0 for %c5120] : i32 -> !stream.resource<transient>{%c5120}
      stream.cmd.dispatch @f_5_dynamic_Nx1280xHxW_1280_1280_Nx1280xHxW_dispatch_0::@cuda_nvptx_fb::@f_5_dynamic_Nx1280xHxW_1280_1280_Nx1280xHxW_dispatch_0_generic_Dx1280xDxD_f32[%0, %1, %2](%10, %11, %12 : i32, i32, i32) {
        ro %arg3[%c0 for %5] : !stream.resource<external>{%5},
        ro %arg4[%c0 for %c5120] : !stream.resource<external>{%c5120},
        ro %arg7[%c0 for %c5120] : !stream.resource<transient>{%c5120},
        ro %arg5[%c0 for %c5120] : !stream.resource<external>{%c5120},
        wo %arg6[%c0 for %5] : !stream.resource<external>{%5}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>, #hal.interface.binding<0, 4>]}
    } => !stream.timepoint
    %14 = stream.resource.dealloca await(%13) => %result : !stream.resource<transient>{%c5120} => !stream.timepoint
    %15 = stream.timepoint.await %14 => %9 : !stream.resource<external>{%5}
    %16 = stream.tensor.export %15 : tensor<?x1280x?x?xf32>{%0, %1, %2} in !stream.resource<external>{%5} -> !hal.buffer_view
    return %16 : !hal.buffer_view
  }
}
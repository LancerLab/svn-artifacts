module @f_10_dynamic_16x512xHxW_512_512_16x512xHxW attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_10_dynamic_16x512xHxW_512_512_16x512xHxW_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_10_dynamic_16x512xHxW_512_512_16x512xHxW_dispatch_0_generic_16x512xDxD_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer, ReadOnly>, <4, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_10_dynamic_16x512xHxW_512_512_16x512xHxW_dispatch_0_generic_16x512xDxD_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = arith.index_castui %0 : i32 to index
          %3 = arith.index_castui %1 : i32 to index
          %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512xf32>>
          %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512xf32>>
          %6 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512xf32>>
          %7 = flow.dispatch.workload.ordinal %2, 0 : index
          %8 = flow.dispatch.workload.ordinal %3, 1 : index
          %9 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x512x?x?xf32>>{%7, %8}
          %10 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x512x?x?xf32>>{%7, %8}
          %11 = flow.dispatch.tensor.load %9, offsets = [0, 0, 0, 0], sizes = [16, 512, %7, %8], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x512x?x?xf32>>{%7, %8} -> tensor<16x512x?x?xf32>
          %12 = flow.dispatch.tensor.load %4, offsets = [0], sizes = [512], strides = [1] : !flow.dispatch.tensor<readonly:tensor<512xf32>> -> tensor<512xf32>
          %13 = flow.dispatch.tensor.load %5, offsets = [0], sizes = [512], strides = [1] : !flow.dispatch.tensor<readonly:tensor<512xf32>> -> tensor<512xf32>
          %14 = flow.dispatch.tensor.load %6, offsets = [0], sizes = [512], strides = [1] : !flow.dispatch.tensor<readonly:tensor<512xf32>> -> tensor<512xf32>
          %15 = tensor.empty(%7, %8) : tensor<16x512x?x?xf32>
          %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%11, %12, %13, %14 : tensor<16x512x?x?xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%15 : tensor<16x512x?x?xf32>) {
          ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
            %17 = arith.mulf %in, %in_0 : f32
            %18 = arith.divf %17, %in_1 : f32
            %19 = arith.addf %18, %in_2 : f32
            linalg.yield %19 : f32
          } -> tensor<16x512x?x?xf32>
          flow.dispatch.tensor.store %16, %10, offsets = [0, 0, 0, 0], sizes = [16, 512, %7, %8], strides = [1, 1, 1, 1] : tensor<16x512x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x512x?x?xf32>>{%7, %8}
          return
        }
      }
    }
  }
  func.func @f_10_dynamic_16x512xHxW_512_512_16x512xHxW(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c2048 = arith.constant 2048 : index
    %c1065353258_i32 = arith.constant 1065353258 : i32
    %c32768 = arith.constant 32768 : index
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c16 = arith.constant 16 : index
    %c1_i32 = arith.constant 1 : i32
    %c553648160_i32 = arith.constant 553648160 : i32
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[2] : index
    %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[3] : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c16, %c512, %0, %1]) type(%c553648160_i32) encoding(%c1_i32)
    %2 = arith.muli %0, %c32768 : index
    %3 = arith.muli %2, %1 : index
    %4 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<16x512x?x?xf32>{%0, %1} in !stream.resource<external>{%3}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c512]) type(%c553648160_i32) encoding(%c1_i32)
    %5 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<512xf32> in !stream.resource<external>{%c2048}
    hal.buffer_view.assert<%arg2 : !hal.buffer_view> message("input 2") shape([%c512]) type(%c553648160_i32) encoding(%c1_i32)
    %6 = stream.tensor.import %arg2 : !hal.buffer_view -> tensor<512xf32> in !stream.resource<external>{%c2048}
    %7 = stream.resource.alloc uninitialized : !stream.resource<external>{%3}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c2048} => !stream.timepoint
    %8 = arith.index_castui %0 : index to i32
    %9 = arith.index_castui %1 : index to i32
    %10 = stream.cmd.execute await(%result_timepoint) => with(%4 as %arg3: !stream.resource<external>{%3}, %5 as %arg4: !stream.resource<external>{%c2048}, %6 as %arg5: !stream.resource<external>{%c2048}, %7 as %arg6: !stream.resource<external>{%3}, %result as %arg7: !stream.resource<transient>{%c2048}) {
      stream.cmd.fill %c1065353258_i32, %arg7[%c0 for %c2048] : i32 -> !stream.resource<transient>{%c2048}
      stream.cmd.dispatch @f_10_dynamic_16x512xHxW_512_512_16x512xHxW_dispatch_0::@cuda_nvptx_fb::@f_10_dynamic_16x512xHxW_512_512_16x512xHxW_dispatch_0_generic_16x512xDxD_f32[%0, %1](%8, %9 : i32, i32) {
        ro %arg3[%c0 for %3] : !stream.resource<external>{%3},
        ro %arg4[%c0 for %c2048] : !stream.resource<external>{%c2048},
        ro %arg7[%c0 for %c2048] : !stream.resource<transient>{%c2048},
        ro %arg5[%c0 for %c2048] : !stream.resource<external>{%c2048},
        wo %arg6[%c0 for %3] : !stream.resource<external>{%3}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>, #hal.interface.binding<0, 4>]}
    } => !stream.timepoint
    %11 = stream.resource.dealloca await(%10) => %result : !stream.resource<transient>{%c2048} => !stream.timepoint
    %12 = stream.timepoint.await %11 => %7 : !stream.resource<external>{%3}
    %13 = stream.tensor.export %12 : tensor<16x512x?x?xf32>{%0, %1} in !stream.resource<external>{%3} -> !hal.buffer_view
    return %13 : !hal.buffer_view
  }
}
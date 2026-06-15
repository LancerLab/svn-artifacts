module @f_10_dynamic_16x512xHxW_16x512xHxW_16x512xHxW attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_10_dynamic_16x512xHxW_16x512xHxW_16x512xHxW_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_10_dynamic_16x512xHxW_16x512xHxW_16x512xHxW_dispatch_0_generic_16x512xDxD_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 4, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3, %arg4
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_10_dynamic_16x512xHxW_16x512xHxW_16x512xHxW_dispatch_0_generic_16x512xDxD_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = hal.interface.constant.load[2] : i32
          %3 = hal.interface.constant.load[3] : i32
          %4 = arith.index_castui %0 : i32 to index
          %5 = arith.index_castui %1 : i32 to index
          %6 = arith.index_castui %2 : i32 to index
          %7 = arith.index_castui %3 : i32 to index
          %8 = flow.dispatch.workload.ordinal %4, 0 : index
          %9 = flow.dispatch.workload.ordinal %5, 1 : index
          %10 = flow.dispatch.workload.ordinal %6, 2 : index
          %11 = flow.dispatch.workload.ordinal %7, 3 : index
          %12 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x512x?x?xf32>>{%10, %11}
          %13 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x512x?x?xf32>>{%8, %9}
          %14 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x512x?x?xf32>>{%10, %11}
          %15 = flow.dispatch.tensor.load %12, offsets = [0, 0, 0, 0], sizes = [16, 512, %10, %11], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x512x?x?xf32>>{%10, %11} -> tensor<16x512x?x?xf32>
          %16 = flow.dispatch.tensor.load %13, offsets = [0, 0, 0, 0], sizes = [16, 512, %8, %9], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x512x?x?xf32>>{%8, %9} -> tensor<16x512x?x?xf32>
          %17 = tensor.empty(%10, %11) : tensor<16x512x?x?xf32>
          %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15, %16 : tensor<16x512x?x?xf32>, tensor<16x512x?x?xf32>) outs(%17 : tensor<16x512x?x?xf32>) {
          ^bb0(%in: f32, %in_0: f32, %out: f32):
            %19 = arith.addf %in, %in_0 : f32
            linalg.yield %19 : f32
          } -> tensor<16x512x?x?xf32>
          flow.dispatch.tensor.store %18, %14, offsets = [0, 0, 0, 0], sizes = [16, 512, %10, %11], strides = [1, 1, 1, 1] : tensor<16x512x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x512x?x?xf32>>{%10, %11}
          return
        }
      }
    }
  }
  func.func @f_10_dynamic_16x512xHxW_16x512xHxW_16x512xHxW(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
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
    %5 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[2] : index
    %6 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[3] : index
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c16, %c512, %5, %6]) type(%c553648160_i32) encoding(%c1_i32)
    %7 = arith.muli %5, %c32768 : index
    %8 = arith.muli %7, %6 : index
    %9 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<16x512x?x?xf32>{%5, %6} in !stream.resource<external>{%8}
    %10 = stream.resource.alloc uninitialized : !stream.resource<external>{%3}
    %11 = arith.index_castui %5 : index to i32
    %12 = arith.index_castui %6 : index to i32
    %13 = arith.index_castui %0 : index to i32
    %14 = arith.index_castui %1 : index to i32
    %15 = stream.cmd.execute with(%4 as %arg2: !stream.resource<external>{%3}, %9 as %arg3: !stream.resource<external>{%8}, %10 as %arg4: !stream.resource<external>{%3}) {
      stream.cmd.dispatch @f_10_dynamic_16x512xHxW_16x512xHxW_16x512xHxW_dispatch_0::@cuda_nvptx_fb::@f_10_dynamic_16x512xHxW_16x512xHxW_16x512xHxW_dispatch_0_generic_16x512xDxD_f32[%5, %6, %0, %1](%11, %12, %13, %14 : i32, i32, i32, i32) {
        ro %arg2[%c0 for %3] : !stream.resource<external>{%3},
        ro %arg3[%c0 for %8] : !stream.resource<external>{%8},
        wo %arg4[%c0 for %3] : !stream.resource<external>{%3}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]}
    } => !stream.timepoint
    %16 = stream.timepoint.await %15 => %10 : !stream.resource<external>{%3}
    %17 = stream.tensor.export %16 : tensor<16x512x?x?xf32>{%0, %1} in !stream.resource<external>{%3} -> !hal.buffer_view
    return %17 : !hal.buffer_view
  }
}
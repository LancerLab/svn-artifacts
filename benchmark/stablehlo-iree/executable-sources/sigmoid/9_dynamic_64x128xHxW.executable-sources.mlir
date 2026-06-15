module @f_9_dynamic_64x128xHxW attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_9_dynamic_64x128xHxW_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_9_dynamic_64x128xHxW_dispatch_0_generic_64x128xDxD_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_9_dynamic_64x128xHxW_dispatch_0_generic_64x128xDxD_f32() {
          %cst = arith.constant 1.000000e+00 : f32
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = arith.index_castui %0 : i32 to index
          %3 = arith.index_castui %1 : i32 to index
          %4 = flow.dispatch.workload.ordinal %2, 0 : index
          %5 = flow.dispatch.workload.ordinal %3, 1 : index
          %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x128x?x?xf32>>{%4, %5}
          %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x128x?x?xf32>>{%4, %5}
          %8 = flow.dispatch.tensor.load %6, offsets = [0, 0, 0, 0], sizes = [64, 128, %4, %5], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x128x?x?xf32>>{%4, %5} -> tensor<64x128x?x?xf32>
          %9 = tensor.empty(%4, %5) : tensor<64x128x?x?xf32>
          %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8 : tensor<64x128x?x?xf32>) outs(%9 : tensor<64x128x?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %11 = arith.negf %in : f32
            %12 = math.exp %11 : f32
            %13 = arith.addf %12, %cst : f32
            %14 = arith.divf %cst, %13 : f32
            linalg.yield %14 : f32
          } -> tensor<64x128x?x?xf32>
          flow.dispatch.tensor.store %10, %7, offsets = [0, 0, 0, 0], sizes = [64, 128, %4, %5], strides = [1, 1, 1, 1] : tensor<64x128x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x128x?x?xf32>>{%4, %5}
          return
        }
      }
    }
  }
  func.func @f_9_dynamic_64x128xHxW(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c32768 = arith.constant 32768 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c1_i32 = arith.constant 1 : i32
    %c553648160_i32 = arith.constant 553648160 : i32
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[2] : index
    %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[3] : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c64, %c128, %0, %1]) type(%c553648160_i32) encoding(%c1_i32)
    %2 = arith.muli %0, %c32768 : index
    %3 = arith.muli %2, %1 : index
    %4 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<64x128x?x?xf32>{%0, %1} in !stream.resource<external>{%3}
    %5 = stream.resource.alloc uninitialized : !stream.resource<external>{%3}
    %6 = arith.index_castui %0 : index to i32
    %7 = arith.index_castui %1 : index to i32
    %8 = stream.cmd.execute with(%4 as %arg1: !stream.resource<external>{%3}, %5 as %arg2: !stream.resource<external>{%3}) {
      stream.cmd.dispatch @f_9_dynamic_64x128xHxW_dispatch_0::@cuda_nvptx_fb::@f_9_dynamic_64x128xHxW_dispatch_0_generic_64x128xDxD_f32[%0, %1](%6, %7 : i32, i32) {
        ro %arg1[%c0 for %3] : !stream.resource<external>{%3},
        wo %arg2[%c0 for %3] : !stream.resource<external>{%3}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %9 = stream.timepoint.await %8 => %5 : !stream.resource<external>{%3}
    %10 = stream.tensor.export %9 : tensor<64x128x?x?xf32>{%0, %1} in !stream.resource<external>{%3} -> !hal.buffer_view
    return %10 : !hal.buffer_view
  }
}
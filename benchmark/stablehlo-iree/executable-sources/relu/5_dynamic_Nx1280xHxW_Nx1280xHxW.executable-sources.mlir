module @f_5_dynamic_Nx1280xHxW_Nx1280xHxW attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_5_dynamic_Nx1280xHxW_Nx1280xHxW_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_5_dynamic_Nx1280xHxW_Nx1280xHxW_dispatch_0_generic_Dx1280xDxD_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 3, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_5_dynamic_Nx1280xHxW_Nx1280xHxW_dispatch_0_generic_Dx1280xDxD_f32() {
          %cst = arith.constant 0.000000e+00 : f32
          %cst_0 = arith.constant 3.40282347E+38 : f32
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = hal.interface.constant.load[2] : i32
          %3 = arith.index_castui %0 : i32 to index
          %4 = arith.index_castui %1 : i32 to index
          %5 = arith.index_castui %2 : i32 to index
          %6 = flow.dispatch.workload.ordinal %3, 0 : index
          %7 = flow.dispatch.workload.ordinal %4, 1 : index
          %8 = flow.dispatch.workload.ordinal %5, 2 : index
          %9 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x1280x?x?xf32>>{%6, %7, %8}
          %10 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x1280x?x?xf32>>{%6, %7, %8}
          %11 = flow.dispatch.tensor.load %9, offsets = [0, 0, 0, 0], sizes = [%6, 1280, %7, %8], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x1280x?x?xf32>>{%6, %7, %8} -> tensor<?x1280x?x?xf32>
          %12 = tensor.empty(%6, %7, %8) : tensor<?x1280x?x?xf32>
          %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%11 : tensor<?x1280x?x?xf32>) outs(%12 : tensor<?x1280x?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %14 = arith.maxf %in, %cst : f32
            %15 = arith.minf %14, %cst_0 : f32
            linalg.yield %15 : f32
          } -> tensor<?x1280x?x?xf32>
          flow.dispatch.tensor.store %13, %10, offsets = [0, 0, 0, 0], sizes = [%6, 1280, %7, %8], strides = [1, 1, 1, 1] : tensor<?x1280x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x1280x?x?xf32>>{%6, %7, %8}
          return
        }
      }
    }
  }
  func.func @f_5_dynamic_Nx1280xHxW_Nx1280xHxW(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c5120 = arith.constant 5120 : index
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
    %7 = stream.resource.alloc uninitialized : !stream.resource<external>{%5}
    %8 = arith.index_castui %0 : index to i32
    %9 = arith.index_castui %1 : index to i32
    %10 = arith.index_castui %2 : index to i32
    %11 = stream.cmd.execute with(%6 as %arg1: !stream.resource<external>{%5}, %7 as %arg2: !stream.resource<external>{%5}) {
      stream.cmd.dispatch @f_5_dynamic_Nx1280xHxW_Nx1280xHxW_dispatch_0::@cuda_nvptx_fb::@f_5_dynamic_Nx1280xHxW_Nx1280xHxW_dispatch_0_generic_Dx1280xDxD_f32[%0, %1, %2](%8, %9, %10 : i32, i32, i32) {
        ro %arg1[%c0 for %5] : !stream.resource<external>{%5},
        wo %arg2[%c0 for %5] : !stream.resource<external>{%5}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %12 = stream.timepoint.await %11 => %7 : !stream.resource<external>{%5}
    %13 = stream.tensor.export %12 : tensor<?x1280x?x?xf32>{%0, %1, %2} in !stream.resource<external>{%5} -> !hal.buffer_view
    return %13 : !hal.buffer_view
  }
}
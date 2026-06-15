module @f_5_dynamic_Nx1280xHxW_Nx320xHxW_Nx1600xHxW attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_5_dynamic_Nx1280xHxW_Nx320xHxW_Nx1600xHxW_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_5_dynamic_Nx1280xHxW_Nx320xHxW_Nx1600xHxW_dispatch_0_generic_Dx1600xDxD_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 6, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3, %arg4, %arg5, %arg6
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_5_dynamic_Nx1280xHxW_Nx320xHxW_Nx1600xHxW_dispatch_0_generic_Dx1600xDxD_f32() {
          %c1280 = arith.constant 1280 : index
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = hal.interface.constant.load[2] : i32
          %3 = hal.interface.constant.load[3] : i32
          %4 = hal.interface.constant.load[4] : i32
          %5 = hal.interface.constant.load[5] : i32
          %6 = arith.index_castui %0 : i32 to index
          %7 = arith.index_castui %1 : i32 to index
          %8 = arith.index_castui %2 : i32 to index
          %9 = arith.index_castui %3 : i32 to index
          %10 = arith.index_castui %4 : i32 to index
          %11 = arith.index_castui %5 : i32 to index
          %12 = flow.dispatch.workload.ordinal %6, 0 : index
          %13 = flow.dispatch.workload.ordinal %7, 1 : index
          %14 = flow.dispatch.workload.ordinal %8, 2 : index
          %15 = flow.dispatch.workload.ordinal %9, 3 : index
          %16 = flow.dispatch.workload.ordinal %10, 4 : index
          %17 = flow.dispatch.workload.ordinal %11, 5 : index
          %18 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x1280x?x?xf32>>{%15, %16, %17}
          %19 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x320x?x?xf32>>{%12, %13, %14}
          %20 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x1600x?x?xf32>>{%15, %16, %17}
          %21 = flow.dispatch.tensor.load %18, offsets = [0, 0, 0, 0], sizes = [%15, 1280, %16, %17], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x1280x?x?xf32>>{%15, %16, %17} -> tensor<?x1280x?x?xf32>
          %22 = flow.dispatch.tensor.load %19, offsets = [0, 0, 0, 0], sizes = [%12, 320, %13, %14], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x320x?x?xf32>>{%12, %13, %14} -> tensor<?x320x?x?xf32>
          %23 = tensor.empty(%15, %16, %17) : tensor<?x1600x?x?xf32>
          %24 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%23 : tensor<?x1600x?x?xf32>) {
          ^bb0(%out: f32):
            %25 = linalg.index 0 : index
            %26 = linalg.index 2 : index
            %27 = linalg.index 3 : index
            %28 = linalg.index 1 : index
            %29 = arith.cmpi ult, %28, %c1280 : index
            %30 = scf.if %29 -> (f32) {
              %extracted = tensor.extract %21[%25, %28, %26, %27] : tensor<?x1280x?x?xf32>
              scf.yield %extracted : f32
            } else {
              %31 = arith.subi %28, %c1280 : index
              %extracted = tensor.extract %22[%25, %31, %26, %27] : tensor<?x320x?x?xf32>
              scf.yield %extracted : f32
            }
            linalg.yield %30 : f32
          } -> tensor<?x1600x?x?xf32>
          flow.dispatch.tensor.store %24, %20, offsets = [0, 0, 0, 0], sizes = [%15, 1600, %16, %17], strides = [1, 1, 1, 1] : tensor<?x1600x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x1600x?x?xf32>>{%15, %16, %17}
          return
        }
      }
    }
  }
  func.func @f_5_dynamic_Nx1280xHxW_Nx320xHxW_Nx1600xHxW(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c5120 = arith.constant 5120 : index
    %c1280 = arith.constant 1280 : index
    %c6400 = arith.constant 6400 : index
    %c0 = arith.constant 0 : index
    %c320 = arith.constant 320 : index
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
    %7 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
    %8 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[2] : index
    %9 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[3] : index
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%7, %c320, %8, %9]) type(%c553648160_i32) encoding(%c1_i32)
    %10 = arith.muli %7, %c1280 : index
    %11 = arith.muli %10, %8 : index
    %12 = arith.muli %11, %9 : index
    %13 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<?x320x?x?xf32>{%7, %8, %9} in !stream.resource<external>{%12}
    %14 = arith.muli %0, %c6400 : index
    %15 = arith.muli %14, %1 : index
    %16 = arith.muli %15, %2 : index
    %17 = stream.resource.alloc uninitialized : !stream.resource<external>{%16}
    %18 = arith.index_castui %7 : index to i32
    %19 = arith.index_castui %8 : index to i32
    %20 = arith.index_castui %9 : index to i32
    %21 = arith.index_castui %0 : index to i32
    %22 = arith.index_castui %1 : index to i32
    %23 = arith.index_castui %2 : index to i32
    %24 = stream.cmd.execute with(%6 as %arg2: !stream.resource<external>{%5}, %13 as %arg3: !stream.resource<external>{%12}, %17 as %arg4: !stream.resource<external>{%16}) {
      stream.cmd.dispatch @f_5_dynamic_Nx1280xHxW_Nx320xHxW_Nx1600xHxW_dispatch_0::@cuda_nvptx_fb::@f_5_dynamic_Nx1280xHxW_Nx320xHxW_Nx1600xHxW_dispatch_0_generic_Dx1600xDxD_f32[%7, %8, %9, %0, %1, %2](%18, %19, %20, %21, %22, %23 : i32, i32, i32, i32, i32, i32) {
        ro %arg2[%c0 for %5] : !stream.resource<external>{%5},
        ro %arg3[%c0 for %12] : !stream.resource<external>{%12},
        wo %arg4[%c0 for %16] : !stream.resource<external>{%16}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]}
    } => !stream.timepoint
    %25 = stream.timepoint.await %24 => %17 : !stream.resource<external>{%16}
    %26 = stream.tensor.export %25 : tensor<?x1600x?x?xf32>{%0, %1, %2} in !stream.resource<external>{%16} -> !hal.buffer_view
    return %26 : !hal.buffer_view
  }
}
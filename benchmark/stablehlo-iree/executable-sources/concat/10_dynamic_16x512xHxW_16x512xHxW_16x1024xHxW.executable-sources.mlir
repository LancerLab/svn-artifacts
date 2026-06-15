module @f_10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW_dispatch_0_generic_16x1024xDxD_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 4, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3, %arg4
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW_dispatch_0_generic_16x1024xDxD_f32() {
          %c512 = arith.constant 512 : index
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
          %14 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x1024x?x?xf32>>{%10, %11}
          %15 = flow.dispatch.tensor.load %12, offsets = [0, 0, 0, 0], sizes = [16, 512, %10, %11], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x512x?x?xf32>>{%10, %11} -> tensor<16x512x?x?xf32>
          %16 = flow.dispatch.tensor.load %13, offsets = [0, 0, 0, 0], sizes = [16, 512, %8, %9], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x512x?x?xf32>>{%8, %9} -> tensor<16x512x?x?xf32>
          %17 = tensor.empty(%10, %11) : tensor<16x1024x?x?xf32>
          %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%17 : tensor<16x1024x?x?xf32>) {
          ^bb0(%out: f32):
            %19 = linalg.index 0 : index
            %20 = linalg.index 2 : index
            %21 = linalg.index 3 : index
            %22 = linalg.index 1 : index
            %23 = arith.cmpi ult, %22, %c512 : index
            %24 = scf.if %23 -> (f32) {
              %extracted = tensor.extract %15[%19, %22, %20, %21] : tensor<16x512x?x?xf32>
              scf.yield %extracted : f32
            } else {
              %25 = arith.subi %22, %c512 : index
              %extracted = tensor.extract %16[%19, %25, %20, %21] : tensor<16x512x?x?xf32>
              scf.yield %extracted : f32
            }
            linalg.yield %24 : f32
          } -> tensor<16x1024x?x?xf32>
          flow.dispatch.tensor.store %18, %14, offsets = [0, 0, 0, 0], sizes = [16, 1024, %10, %11], strides = [1, 1, 1, 1] : tensor<16x1024x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x1024x?x?xf32>>{%10, %11}
          return
        }
      }
    }
  }
  func.func @f_10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c32768 = arith.constant 32768 : index
    %c65536 = arith.constant 65536 : index
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
    %10 = arith.muli %0, %c65536 : index
    %11 = arith.muli %10, %1 : index
    %12 = stream.resource.alloc uninitialized : !stream.resource<external>{%11}
    %13 = arith.index_castui %5 : index to i32
    %14 = arith.index_castui %6 : index to i32
    %15 = arith.index_castui %0 : index to i32
    %16 = arith.index_castui %1 : index to i32
    %17 = stream.cmd.execute with(%4 as %arg2: !stream.resource<external>{%3}, %9 as %arg3: !stream.resource<external>{%8}, %12 as %arg4: !stream.resource<external>{%11}) {
      stream.cmd.dispatch @f_10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW_dispatch_0::@cuda_nvptx_fb::@f_10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW_dispatch_0_generic_16x1024xDxD_f32[%5, %6, %0, %1](%13, %14, %15, %16 : i32, i32, i32, i32) {
        ro %arg2[%c0 for %3] : !stream.resource<external>{%3},
        ro %arg3[%c0 for %8] : !stream.resource<external>{%8},
        wo %arg4[%c0 for %11] : !stream.resource<external>{%11}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]}
    } => !stream.timepoint
    %18 = stream.timepoint.await %17 => %12 : !stream.resource<external>{%11}
    %19 = stream.tensor.export %18 : tensor<16x1024x?x?xf32>{%0, %1} in !stream.resource<external>{%11} -> !hal.buffer_view
    return %19 : !hal.buffer_view
  }
}
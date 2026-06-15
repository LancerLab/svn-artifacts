module @f_5_dynamic_Bx2048_2048x1000_Bx1000 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_5_dynamic_Bx2048_2048x1000_Bx1000_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_5_dynamic_Bx2048_2048x1000_Bx1000_dispatch_0_matmul_Dx1000x2048_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 1, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_5_dynamic_Bx2048_2048x1000_Bx1000_dispatch_0_matmul_Dx1000x2048_f32() {
          %cst = arith.constant 0.000000e+00 : f32
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = arith.index_castui %0 : i32 to index
          %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1000xf32>>
          %3 = flow.dispatch.workload.ordinal %1, 0 : index
          %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x2048xf32>>{%3}
          %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x1000xf32>>{%3}
          %6 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%3, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x2048xf32>>{%3} -> tensor<?x2048xf32>
          %7 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [2048, 1000], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1000xf32>> -> tensor<2048x1000xf32>
          %8 = tensor.empty(%3) : tensor<?x1000xf32>
          %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<?x1000xf32>) -> tensor<?x1000xf32>
          %10 = linalg.matmul ins(%6, %7 : tensor<?x2048xf32>, tensor<2048x1000xf32>) outs(%9 : tensor<?x1000xf32>) -> tensor<?x1000xf32>
          flow.dispatch.tensor.store %10, %5, offsets = [0, 0], sizes = [%3, 1000], strides = [1, 1] : tensor<?x1000xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x1000xf32>>{%3}
          return
        }
      }
    }
  }
  func.func @f_5_dynamic_Bx2048_2048x1000_Bx1000(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c8192 = arith.constant 8192 : index
    %c8192000 = arith.constant 8192000 : index
    %c4000 = arith.constant 4000 : index
    %c0 = arith.constant 0 : index
    %c1000 = arith.constant 1000 : index
    %c2048 = arith.constant 2048 : index
    %c1_i32 = arith.constant 1 : i32
    %c553648160_i32 = arith.constant 553648160 : i32
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%0, %c2048]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = arith.muli %0, %c8192 : index
    %2 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<?x2048xf32>{%0} in !stream.resource<external>{%1}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c2048, %c1000]) type(%c553648160_i32) encoding(%c1_i32)
    %3 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<2048x1000xf32> in !stream.resource<external>{%c8192000}
    %4 = arith.muli %0, %c4000 : index
    %5 = stream.resource.alloc uninitialized : !stream.resource<external>{%4}
    %6 = arith.index_castui %0 : index to i32
    %7 = stream.cmd.execute with(%2 as %arg2: !stream.resource<external>{%1}, %3 as %arg3: !stream.resource<external>{%c8192000}, %5 as %arg4: !stream.resource<external>{%4}) {
      stream.cmd.dispatch @f_5_dynamic_Bx2048_2048x1000_Bx1000_dispatch_0::@cuda_nvptx_fb::@f_5_dynamic_Bx2048_2048x1000_Bx1000_dispatch_0_matmul_Dx1000x2048_f32[%0](%6 : i32) {
        ro %arg2[%c0 for %1] : !stream.resource<external>{%1},
        ro %arg3[%c0 for %c8192000] : !stream.resource<external>{%c8192000},
        wo %arg4[%c0 for %4] : !stream.resource<external>{%4}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]}
    } => !stream.timepoint
    %8 = stream.timepoint.await %7 => %5 : !stream.resource<external>{%4}
    %9 = stream.tensor.export %8 : tensor<?x1000xf32>{%0} in !stream.resource<external>{%4} -> !hal.buffer_view
    return %9 : !hal.buffer_view
  }
}
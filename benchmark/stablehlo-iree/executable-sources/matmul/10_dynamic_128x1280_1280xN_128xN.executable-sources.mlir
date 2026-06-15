module @f_10_dynamic_128x1280_1280xN_128xN attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_10_dynamic_128x1280_1280xN_128xN_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_10_dynamic_128x1280_1280xN_128xN_dispatch_0_matmul_128xDx1280_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 1, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_10_dynamic_128x1280_1280xN_128xN_dispatch_0_matmul_128xDx1280_f32() {
          %cst = arith.constant 0.000000e+00 : f32
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = arith.index_castui %0 : i32 to index
          %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x1280xf32>>
          %3 = flow.dispatch.workload.ordinal %1, 0 : index
          %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280x?xf32>>{%3}
          %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x?xf32>>{%3}
          %6 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [128, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x1280xf32>> -> tensor<128x1280xf32>
          %7 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [1280, %3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1280x?xf32>>{%3} -> tensor<1280x?xf32>
          %8 = tensor.empty(%3) : tensor<128x?xf32>
          %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<128x?xf32>) -> tensor<128x?xf32>
          %10 = linalg.matmul ins(%6, %7 : tensor<128x1280xf32>, tensor<1280x?xf32>) outs(%9 : tensor<128x?xf32>) -> tensor<128x?xf32>
          flow.dispatch.tensor.store %10, %5, offsets = [0, 0], sizes = [128, %3], strides = [1, 1] : tensor<128x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x?xf32>>{%3}
          return
        }
      }
    }
  }
  func.func @f_10_dynamic_128x1280_1280xN_128xN(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c655360 = arith.constant 655360 : index
    %c5120 = arith.constant 5120 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128 = arith.constant 128 : index
    %c1280 = arith.constant 1280 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c128, %c1280]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<128x1280xf32> in !stream.resource<external>{%c655360}
    %1 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c1280, %1]) type(%c553648160_i32) encoding(%c1_i32)
    %2 = arith.muli %1, %c5120 : index
    %3 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<1280x?xf32>{%1} in !stream.resource<external>{%2}
    %4 = arith.muli %1, %c512 : index
    %5 = stream.resource.alloc uninitialized : !stream.resource<external>{%4}
    %6 = arith.index_castui %1 : index to i32
    %7 = stream.cmd.execute with(%0 as %arg2: !stream.resource<external>{%c655360}, %3 as %arg3: !stream.resource<external>{%2}, %5 as %arg4: !stream.resource<external>{%4}) {
      stream.cmd.dispatch @f_10_dynamic_128x1280_1280xN_128xN_dispatch_0::@cuda_nvptx_fb::@f_10_dynamic_128x1280_1280xN_128xN_dispatch_0_matmul_128xDx1280_f32[%1](%6 : i32) {
        ro %arg2[%c0 for %c655360] : !stream.resource<external>{%c655360},
        ro %arg3[%c0 for %2] : !stream.resource<external>{%2},
        wo %arg4[%c0 for %4] : !stream.resource<external>{%4}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]}
    } => !stream.timepoint
    %8 = stream.timepoint.await %7 => %5 : !stream.resource<external>{%4}
    %9 = stream.tensor.export %8 : tensor<128x?xf32>{%1} in !stream.resource<external>{%4} -> !hal.buffer_view
    return %9 : !hal.buffer_view
  }
}
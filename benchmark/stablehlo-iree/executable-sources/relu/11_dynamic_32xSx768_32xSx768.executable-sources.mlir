module @f_11_dynamic_32xSx768_32xSx768 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_11_dynamic_32xSx768_32xSx768_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_11_dynamic_32xSx768_32xSx768_dispatch_0_generic_32xDx768_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 1, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_11_dynamic_32xSx768_32xSx768_dispatch_0_generic_32xDx768_f32() {
          %cst = arith.constant 0.000000e+00 : f32
          %cst_0 = arith.constant 3.40282347E+38 : f32
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = arith.index_castui %0 : i32 to index
          %2 = flow.dispatch.workload.ordinal %1, 0 : index
          %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x?x768xf32>>{%2}
          %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x?x768xf32>>{%2}
          %5 = flow.dispatch.tensor.load %3, offsets = [0, 0, 0], sizes = [32, %2, 768], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x?x768xf32>>{%2} -> tensor<32x?x768xf32>
          %6 = tensor.empty(%2) : tensor<32x?x768xf32>
          %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<32x?x768xf32>) outs(%6 : tensor<32x?x768xf32>) {
          ^bb0(%in: f32, %out: f32):
            %8 = arith.maxf %in, %cst : f32
            %9 = arith.minf %8, %cst_0 : f32
            linalg.yield %9 : f32
          } -> tensor<32x?x768xf32>
          flow.dispatch.tensor.store %7, %4, offsets = [0, 0, 0], sizes = [32, %2, 768], strides = [1, 1, 1] : tensor<32x?x768xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x?x768xf32>>{%2}
          return
        }
      }
    }
  }
  func.func @f_11_dynamic_32xSx768_32xSx768(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c98304 = arith.constant 98304 : index
    %c0 = arith.constant 0 : index
    %c768 = arith.constant 768 : index
    %c32 = arith.constant 32 : index
    %c1_i32 = arith.constant 1 : i32
    %c553648160_i32 = arith.constant 553648160 : i32
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c32, %0, %c768]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = arith.muli %0, %c98304 : index
    %2 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<32x?x768xf32>{%0} in !stream.resource<external>{%1}
    %3 = stream.resource.alloc uninitialized : !stream.resource<external>{%1}
    %4 = arith.index_castui %0 : index to i32
    %5 = stream.cmd.execute with(%2 as %arg1: !stream.resource<external>{%1}, %3 as %arg2: !stream.resource<external>{%1}) {
      stream.cmd.dispatch @f_11_dynamic_32xSx768_32xSx768_dispatch_0::@cuda_nvptx_fb::@f_11_dynamic_32xSx768_32xSx768_dispatch_0_generic_32xDx768_f32[%0](%4 : i32) {
        ro %arg1[%c0 for %1] : !stream.resource<external>{%1},
        wo %arg2[%c0 for %1] : !stream.resource<external>{%1}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %6 = stream.timepoint.await %5 => %3 : !stream.resource<external>{%1}
    %7 = stream.tensor.export %6 : tensor<32x?x768xf32>{%0} in !stream.resource<external>{%1} -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
}
module @f_2_cnn_128x128x28x28_128x128x14x14 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_2_cnn_128x128x28x28_128x128x14x14_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_2_cnn_128x128x28x28_128x128x14x14_dispatch_0_generic_128x128x14x14x2x2_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_2_cnn_128x128x28x28_128x128x14x14_dispatch_0_generic_128x128x14x14x2x2_f32() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant -3.402820e+38 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x128x28x28xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x128x14x14xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [128, 128, 28, 28], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x128x28x28xf32>> -> tensor<128x128x28x28xf32>
          %3 = tensor.empty() : tensor<2x2xf32>
          %4 = tensor.empty() : tensor<128x128x14x14xf32>
          %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%4 : tensor<128x128x14x14xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst : f32
          } -> tensor<128x128x14x14xf32>
          %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d4, d3 * 2 + d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%2, %3 : tensor<128x128x28x28xf32>, tensor<2x2xf32>) outs(%5 : tensor<128x128x14x14xf32>) {
          ^bb0(%in: f32, %in_0: f32, %out: f32):
            %7 = arith.maxf %out, %in : f32
            linalg.yield %7 : f32
          } -> tensor<128x128x14x14xf32>
          flow.dispatch.tensor.store %6, %1, offsets = [0, 0, 0, 0], sizes = [128, 128, 14, 14], strides = [1, 1, 1, 1] : tensor<128x128x14x14xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x128x14x14xf32>>
          return
        }
      }
    }
  }
  func.func @f_2_cnn_128x128x28x28_128x128x14x14(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c51380224 = arith.constant 51380224 : index
    %c12845056 = arith.constant 12845056 : index
    %c0 = arith.constant 0 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128 = arith.constant 128 : index
    %c28 = arith.constant 28 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c128, %c128, %c28, %c28]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<128x128x28x28xf32> in !stream.resource<external>{%c51380224}
    %1 = stream.resource.alloc uninitialized : !stream.resource<external>{%c12845056}
    %2 = stream.cmd.execute with(%0 as %arg1: !stream.resource<external>{%c51380224}, %1 as %arg2: !stream.resource<external>{%c12845056}) {
      stream.cmd.dispatch @f_2_cnn_128x128x28x28_128x128x14x14_dispatch_0::@cuda_nvptx_fb::@f_2_cnn_128x128x28x28_128x128x14x14_dispatch_0_generic_128x128x14x14x2x2_f32 {
        ro %arg1[%c0 for %c51380224] : !stream.resource<external>{%c51380224},
        wo %arg2[%c0 for %c12845056] : !stream.resource<external>{%c12845056}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %3 = stream.timepoint.await %2 => %1 : !stream.resource<external>{%c12845056}
    %4 = stream.tensor.export %3 : tensor<128x128x14x14xf32> in !stream.resource<external>{%c12845056} -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
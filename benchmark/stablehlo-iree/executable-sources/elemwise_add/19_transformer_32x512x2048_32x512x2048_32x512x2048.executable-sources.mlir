module @f_19_transformer_32x512x2048_32x512x2048_32x512x2048 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_19_transformer_32x512x2048_32x512x2048_32x512x2048_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_19_transformer_32x512x2048_32x512x2048_32x512x2048_dispatch_0_generic_33554432_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_19_transformer_32x512x2048_32x512x2048_32x512x2048_dispatch_0_generic_33554432_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<33554432xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<33554432xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<33554432xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [33554432], strides = [1] : !flow.dispatch.tensor<readonly:tensor<33554432xf32>> -> tensor<33554432xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [33554432], strides = [1] : !flow.dispatch.tensor<readonly:tensor<33554432xf32>> -> tensor<33554432xf32>
          %5 = tensor.empty() : tensor<33554432xf32>
          %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3, %4 : tensor<33554432xf32>, tensor<33554432xf32>) outs(%5 : tensor<33554432xf32>) {
          ^bb0(%in: f32, %in_0: f32, %out: f32):
            %7 = arith.addf %in, %in_0 : f32
            linalg.yield %7 : f32
          } -> tensor<33554432xf32>
          flow.dispatch.tensor.store %6, %2, offsets = [0], sizes = [33554432], strides = [1] : tensor<33554432xf32> -> !flow.dispatch.tensor<writeonly:tensor<33554432xf32>>
          return
        }
      }
    }
  }
  func.func @f_19_transformer_32x512x2048_32x512x2048_32x512x2048(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c134217728 = arith.constant 134217728 : index
    %c0 = arith.constant 0 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c2048 = arith.constant 2048 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c32, %c512, %c2048]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<32x512x2048xf32> in !stream.resource<external>{%c134217728}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c32, %c512, %c2048]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<32x512x2048xf32> in !stream.resource<external>{%c134217728}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c134217728}
    %3 = stream.cmd.execute with(%0 as %arg2: !stream.resource<external>{%c134217728}, %1 as %arg3: !stream.resource<external>{%c134217728}, %2 as %arg4: !stream.resource<external>{%c134217728}) {
      stream.cmd.dispatch @f_19_transformer_32x512x2048_32x512x2048_32x512x2048_dispatch_0::@cuda_nvptx_fb::@f_19_transformer_32x512x2048_32x512x2048_32x512x2048_dispatch_0_generic_33554432_f32 {
        ro %arg2[%c0 for %c134217728] : !stream.resource<external>{%c134217728},
        ro %arg3[%c0 for %c134217728] : !stream.resource<external>{%c134217728},
        wo %arg4[%c0 for %c134217728] : !stream.resource<external>{%c134217728}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]}
    } => !stream.timepoint
    %4 = stream.timepoint.await %3 => %2 : !stream.resource<external>{%c134217728}
    %5 = stream.tensor.export %4 : tensor<32x512x2048xf32> in !stream.resource<external>{%c134217728} -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
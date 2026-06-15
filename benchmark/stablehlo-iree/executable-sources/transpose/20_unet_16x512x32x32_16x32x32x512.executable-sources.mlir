module @f_20_unet_16x512x32x32_16x32x32x512 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_20_unet_16x512x32x32_16x32x32x512_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_20_unet_16x512x32x32_16x32x32x512_dispatch_0_generic_16x1024x512_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_20_unet_16x512x32x32_16x32x32x512_dispatch_0_generic_16x1024x512_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x512x1024xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x1024x512xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 512, 1024], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x512x1024xf32>> -> tensor<16x512x1024xf32>
          %3 = tensor.empty() : tensor<16x1024x512xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<16x512x1024xf32>) outs(%3 : tensor<16x1024x512xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<16x1024x512xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [16, 1024, 512], strides = [1, 1, 1] : tensor<16x1024x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x1024x512xf32>>
          return
        }
      }
    }
  }
  func.func @f_20_unet_16x512x32x32_16x32x32x512(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c33554432 = arith.constant 33554432 : index
    %c0 = arith.constant 0 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index
    %c32 = arith.constant 32 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c16, %c512, %c32, %c32]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<16x512x32x32xf32> in !stream.resource<external>{%c33554432}
    %1 = stream.resource.alloc uninitialized : !stream.resource<external>{%c33554432}
    %2 = stream.cmd.execute with(%0 as %arg1: !stream.resource<external>{%c33554432}, %1 as %arg2: !stream.resource<external>{%c33554432}) {
      stream.cmd.dispatch @f_20_unet_16x512x32x32_16x32x32x512_dispatch_0::@cuda_nvptx_fb::@f_20_unet_16x512x32x32_16x32x32x512_dispatch_0_generic_16x1024x512_f32 {
        ro %arg1[%c0 for %c33554432] : !stream.resource<external>{%c33554432},
        wo %arg2[%c0 for %c33554432] : !stream.resource<external>{%c33554432}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %3 = stream.timepoint.await %2 => %1 : !stream.resource<external>{%c33554432}
    %4 = stream.tensor.export %3 : tensor<16x32x32x512xf32> in !stream.resource<external>{%c33554432} -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
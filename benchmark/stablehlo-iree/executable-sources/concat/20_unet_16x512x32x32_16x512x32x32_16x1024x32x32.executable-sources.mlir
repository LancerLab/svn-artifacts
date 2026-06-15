module @f_20_unet_16x512x32x32_16x512x32x32_16x1024x32x32 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_20_unet_16x512x32x32_16x512x32x32_16x1024x32x32_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_20_unet_16x512x32x32_16x512x32x32_16x1024x32x32_dispatch_0 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_20_unet_16x512x32x32_16x512x32x32_16x1024x32x32_dispatch_0() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x512x32x32xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<16x1024x32x32xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [16, 512, 32, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x512x32x32xf32>> -> tensor<16x512x32x32xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 0, 0, 0], sizes = [16, 512, 32, 32], strides = [1, 1, 1, 1] : tensor<16x512x32x32xf32> -> !flow.dispatch.tensor<readwrite:tensor<16x1024x32x32xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_20_unet_16x512x32x32_16x512x32x32_16x1024x32x32_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_20_unet_16x512x32x32_16x512x32x32_16x1024x32x32_dispatch_1 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_20_unet_16x512x32x32_16x512x32x32_16x1024x32x32_dispatch_1() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x512x32x32xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<16x1024x32x32xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [16, 512, 32, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x512x32x32xf32>> -> tensor<16x512x32x32xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 512, 0, 0], sizes = [16, 512, 32, 32], strides = [1, 1, 1, 1] : tensor<16x512x32x32xf32> -> !flow.dispatch.tensor<readwrite:tensor<16x1024x32x32xf32>>
          return
        }
      }
    }
  }
  func.func @f_20_unet_16x512x32x32_16x512x32x32_16x1024x32x32(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c33554432 = arith.constant 33554432 : index
    %c67108864 = arith.constant 67108864 : index
    %c0 = arith.constant 0 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index
    %c32 = arith.constant 32 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c16, %c512, %c32, %c32]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<16x512x32x32xf32> in !stream.resource<external>{%c33554432}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c16, %c512, %c32, %c32]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<16x512x32x32xf32> in !stream.resource<external>{%c33554432}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c67108864}
    %3 = stream.cmd.execute with(%0 as %arg2: !stream.resource<external>{%c33554432}, %1 as %arg3: !stream.resource<external>{%c33554432}, %2 as %arg4: !stream.resource<external>{%c67108864}) {
      stream.cmd.dispatch @f_20_unet_16x512x32x32_16x512x32x32_16x1024x32x32_dispatch_0::@cuda_nvptx_fb::@f_20_unet_16x512x32x32_16x512x32x32_16x1024x32x32_dispatch_0 {
        ro %arg2[%c0 for %c33554432] : !stream.resource<external>{%c33554432},
        rw %arg4[%c0 for %c67108864] : !stream.resource<external>{%c67108864}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_20_unet_16x512x32x32_16x512x32x32_16x1024x32x32_dispatch_1::@cuda_nvptx_fb::@f_20_unet_16x512x32x32_16x512x32x32_16x1024x32x32_dispatch_1 {
        ro %arg3[%c0 for %c33554432] : !stream.resource<external>{%c33554432},
        rw %arg4[%c0 for %c67108864] : !stream.resource<external>{%c67108864}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %4 = stream.timepoint.await %3 => %2 : !stream.resource<external>{%c67108864}
    %5 = stream.tensor.export %4 : tensor<16x1024x32x32xf32> in !stream.resource<external>{%c67108864} -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
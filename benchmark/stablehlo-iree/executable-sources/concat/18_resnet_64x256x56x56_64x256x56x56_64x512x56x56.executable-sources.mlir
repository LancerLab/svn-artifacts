module @f_18_resnet_64x256x56x56_64x256x56x56_64x512x56x56 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_18_resnet_64x256x56x56_64x256x56x56_64x512x56x56_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_18_resnet_64x256x56x56_64x256x56x56_64x512x56x56_dispatch_0 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_18_resnet_64x256x56x56_64x256x56x56_64x512x56x56_dispatch_0() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x256x56x56xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<64x512x56x56xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [64, 256, 56, 56], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x256x56x56xf32>> -> tensor<64x256x56x56xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 0, 0, 0], sizes = [64, 256, 56, 56], strides = [1, 1, 1, 1] : tensor<64x256x56x56xf32> -> !flow.dispatch.tensor<readwrite:tensor<64x512x56x56xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_18_resnet_64x256x56x56_64x256x56x56_64x512x56x56_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_18_resnet_64x256x56x56_64x256x56x56_64x512x56x56_dispatch_1 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_18_resnet_64x256x56x56_64x256x56x56_64x512x56x56_dispatch_1() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x256x56x56xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<64x512x56x56xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [64, 256, 56, 56], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x256x56x56xf32>> -> tensor<64x256x56x56xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 256, 0, 0], sizes = [64, 256, 56, 56], strides = [1, 1, 1, 1] : tensor<64x256x56x56xf32> -> !flow.dispatch.tensor<readwrite:tensor<64x512x56x56xf32>>
          return
        }
      }
    }
  }
  func.func @f_18_resnet_64x256x56x56_64x256x56x56_64x512x56x56(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c205520896 = arith.constant 205520896 : index
    %c411041792 = arith.constant 411041792 : index
    %c0 = arith.constant 0 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c56 = arith.constant 56 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c64, %c256, %c56, %c56]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<64x256x56x56xf32> in !stream.resource<external>{%c205520896}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c64, %c256, %c56, %c56]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<64x256x56x56xf32> in !stream.resource<external>{%c205520896}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c411041792}
    %3 = stream.cmd.execute with(%0 as %arg2: !stream.resource<external>{%c205520896}, %1 as %arg3: !stream.resource<external>{%c205520896}, %2 as %arg4: !stream.resource<external>{%c411041792}) {
      stream.cmd.dispatch @f_18_resnet_64x256x56x56_64x256x56x56_64x512x56x56_dispatch_0::@cuda_nvptx_fb::@f_18_resnet_64x256x56x56_64x256x56x56_64x512x56x56_dispatch_0 {
        ro %arg2[%c0 for %c205520896] : !stream.resource<external>{%c205520896},
        rw %arg4[%c0 for %c411041792] : !stream.resource<external>{%c411041792}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_18_resnet_64x256x56x56_64x256x56x56_64x512x56x56_dispatch_1::@cuda_nvptx_fb::@f_18_resnet_64x256x56x56_64x256x56x56_64x512x56x56_dispatch_1 {
        ro %arg3[%c0 for %c205520896] : !stream.resource<external>{%c205520896},
        rw %arg4[%c0 for %c411041792] : !stream.resource<external>{%c411041792}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %4 = stream.timepoint.await %3 => %2 : !stream.resource<external>{%c411041792}
    %5 = stream.tensor.export %4 : tensor<64x512x56x56xf32> in !stream.resource<external>{%c411041792} -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
module @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_dispatch_0 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_dispatch_0() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x1280x7x7xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<64x1600x7x7xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [64, 1280, 7, 7], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x1280x7x7xf32>> -> tensor<64x1280x7x7xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 0, 0, 0], sizes = [64, 1280, 7, 7], strides = [1, 1, 1, 1] : tensor<64x1280x7x7xf32> -> !flow.dispatch.tensor<readwrite:tensor<64x1600x7x7xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_dispatch_1 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_dispatch_1() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x320x7x7xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<64x1600x7x7xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [64, 320, 7, 7], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x320x7x7xf32>> -> tensor<64x320x7x7xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 1280, 0, 0], sizes = [64, 320, 7, 7], strides = [1, 1, 1, 1] : tensor<64x320x7x7xf32> -> !flow.dispatch.tensor<readwrite:tensor<64x1600x7x7xf32>>
          return
        }
      }
    }
  }
  func.func @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c16056320 = arith.constant 16056320 : index
    %c4014080 = arith.constant 4014080 : index
    %c20070400 = arith.constant 20070400 : index
    %c0 = arith.constant 0 : index
    %c320 = arith.constant 320 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c1280 = arith.constant 1280 : index
    %c7 = arith.constant 7 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c64, %c1280, %c7, %c7]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<64x1280x7x7xf32> in !stream.resource<external>{%c16056320}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c64, %c320, %c7, %c7]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<64x320x7x7xf32> in !stream.resource<external>{%c4014080}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c20070400}
    %3 = stream.cmd.execute with(%0 as %arg2: !stream.resource<external>{%c16056320}, %1 as %arg3: !stream.resource<external>{%c4014080}, %2 as %arg4: !stream.resource<external>{%c20070400}) {
      stream.cmd.dispatch @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_dispatch_0::@cuda_nvptx_fb::@f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_dispatch_0 {
        ro %arg2[%c0 for %c16056320] : !stream.resource<external>{%c16056320},
        rw %arg4[%c0 for %c20070400] : !stream.resource<external>{%c20070400}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_dispatch_1::@cuda_nvptx_fb::@f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_dispatch_1 {
        ro %arg3[%c0 for %c4014080] : !stream.resource<external>{%c4014080},
        rw %arg4[%c0 for %c20070400] : !stream.resource<external>{%c20070400}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %4 = stream.timepoint.await %3 => %2 : !stream.resource<external>{%c20070400}
    %5 = stream.tensor.export %4 : tensor<64x1600x7x7xf32> in !stream.resource<external>{%c20070400} -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
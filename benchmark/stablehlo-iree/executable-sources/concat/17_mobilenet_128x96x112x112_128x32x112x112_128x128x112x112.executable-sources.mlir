module @f_17_mobilenet_128x96x112x112_128x32x112x112_128x128x112x112 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_17_mobilenet_128x96x112x112_128x32x112x112_128x128x112x112_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_17_mobilenet_128x96x112x112_128x32x112x112_128x128x112x112_dispatch_0 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_17_mobilenet_128x96x112x112_128x32x112x112_128x128x112x112_dispatch_0() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x96x112x112xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<128x128x112x112xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [128, 96, 112, 112], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x96x112x112xf32>> -> tensor<128x96x112x112xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 0, 0, 0], sizes = [128, 96, 112, 112], strides = [1, 1, 1, 1] : tensor<128x96x112x112xf32> -> !flow.dispatch.tensor<readwrite:tensor<128x128x112x112xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_17_mobilenet_128x96x112x112_128x32x112x112_128x128x112x112_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_17_mobilenet_128x96x112x112_128x32x112x112_128x128x112x112_dispatch_1 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_17_mobilenet_128x96x112x112_128x32x112x112_128x128x112x112_dispatch_1() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x32x112x112xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<128x128x112x112xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [128, 32, 112, 112], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x32x112x112xf32>> -> tensor<128x32x112x112xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 96, 0, 0], sizes = [128, 32, 112, 112], strides = [1, 1, 1, 1] : tensor<128x32x112x112xf32> -> !flow.dispatch.tensor<readwrite:tensor<128x128x112x112xf32>>
          return
        }
      }
    }
  }
  func.func @f_17_mobilenet_128x96x112x112_128x32x112x112_128x128x112x112(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c616562688 = arith.constant 616562688 : index
    %c205520896 = arith.constant 205520896 : index
    %c822083584 = arith.constant 822083584 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128 = arith.constant 128 : index
    %c96 = arith.constant 96 : index
    %c112 = arith.constant 112 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c128, %c96, %c112, %c112]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<128x96x112x112xf32> in !stream.resource<external>{%c616562688}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c128, %c32, %c112, %c112]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<128x32x112x112xf32> in !stream.resource<external>{%c205520896}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c822083584}
    %3 = stream.cmd.execute with(%0 as %arg2: !stream.resource<external>{%c616562688}, %1 as %arg3: !stream.resource<external>{%c205520896}, %2 as %arg4: !stream.resource<external>{%c822083584}) {
      stream.cmd.dispatch @f_17_mobilenet_128x96x112x112_128x32x112x112_128x128x112x112_dispatch_0::@cuda_nvptx_fb::@f_17_mobilenet_128x96x112x112_128x32x112x112_128x128x112x112_dispatch_0 {
        ro %arg2[%c0 for %c616562688] : !stream.resource<external>{%c616562688},
        rw %arg4[%c0 for %c822083584] : !stream.resource<external>{%c822083584}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_17_mobilenet_128x96x112x112_128x32x112x112_128x128x112x112_dispatch_1::@cuda_nvptx_fb::@f_17_mobilenet_128x96x112x112_128x32x112x112_128x128x112x112_dispatch_1 {
        ro %arg3[%c0 for %c205520896] : !stream.resource<external>{%c205520896},
        rw %arg4[%c0 for %c822083584] : !stream.resource<external>{%c822083584}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %4 = stream.timepoint.await %3 => %2 : !stream.resource<external>{%c822083584}
    %5 = stream.tensor.export %4 : tensor<128x128x112x112xf32> in !stream.resource<external>{%c822083584} -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
module @f_21_vit_32x196x768_32x1x768_32x197x768 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_21_vit_32x196x768_32x1x768_32x197x768_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_21_vit_32x196x768_32x1x768_32x197x768_dispatch_0 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_21_vit_32x196x768_32x1x768_32x197x768_dispatch_0() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x196x768xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<32x197x768xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [32, 196, 768], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x196x768xf32>> -> tensor<32x196x768xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 0, 0], sizes = [32, 196, 768], strides = [1, 1, 1] : tensor<32x196x768xf32> -> !flow.dispatch.tensor<readwrite:tensor<32x197x768xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_21_vit_32x196x768_32x1x768_32x197x768_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_21_vit_32x196x768_32x1x768_32x197x768_dispatch_1 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_21_vit_32x196x768_32x1x768_32x197x768_dispatch_1() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x768xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<32x197x768xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x768xf32>> -> tensor<32x768xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 196, 0], sizes = [32, 1, 768], strides = [1, 1, 1] : tensor<32x768xf32> -> !flow.dispatch.tensor<readwrite:tensor<32x197x768xf32>>
          return
        }
      }
    }
  }
  func.func @f_21_vit_32x196x768_32x1x768_32x197x768(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c19267584 = arith.constant 19267584 : index
    %c98304 = arith.constant 98304 : index
    %c19365888 = arith.constant 19365888 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32 = arith.constant 32 : index
    %c196 = arith.constant 196 : index
    %c768 = arith.constant 768 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c32, %c196, %c768]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<32x196x768xf32> in !stream.resource<external>{%c19267584}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c32, %c1, %c768]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<32x1x768xf32> in !stream.resource<external>{%c98304}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c19365888}
    %3 = stream.cmd.execute with(%0 as %arg2: !stream.resource<external>{%c19267584}, %1 as %arg3: !stream.resource<external>{%c98304}, %2 as %arg4: !stream.resource<external>{%c19365888}) {
      stream.cmd.dispatch @f_21_vit_32x196x768_32x1x768_32x197x768_dispatch_0::@cuda_nvptx_fb::@f_21_vit_32x196x768_32x1x768_32x197x768_dispatch_0 {
        ro %arg2[%c0 for %c19267584] : !stream.resource<external>{%c19267584},
        rw %arg4[%c0 for %c19365888] : !stream.resource<external>{%c19365888}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_21_vit_32x196x768_32x1x768_32x197x768_dispatch_1::@cuda_nvptx_fb::@f_21_vit_32x196x768_32x1x768_32x197x768_dispatch_1 {
        ro %arg3[%c0 for %c98304] : !stream.resource<external>{%c98304},
        rw %arg4[%c0 for %c19365888] : !stream.resource<external>{%c19365888}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %4 = stream.timepoint.await %3 => %2 : !stream.resource<external>{%c19365888}
    %5 = stream.tensor.export %4 : tensor<32x197x768xf32> in !stream.resource<external>{%c19365888} -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
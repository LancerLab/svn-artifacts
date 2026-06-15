module @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_0 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_0() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x512x64xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<32x512x256xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [32, 512, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x512x64xf32>> -> tensor<32x512x64xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 0, 0], sizes = [32, 512, 64], strides = [1, 1, 1] : tensor<32x512x64xf32> -> !flow.dispatch.tensor<readwrite:tensor<32x512x256xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_1 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_1() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x512x64xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<32x512x256xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [32, 512, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x512x64xf32>> -> tensor<32x512x64xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 0, 64], sizes = [32, 512, 64], strides = [1, 1, 1] : tensor<32x512x64xf32> -> !flow.dispatch.tensor<readwrite:tensor<32x512x256xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_2 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_2 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_2() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x512x64xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<32x512x256xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [32, 512, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x512x64xf32>> -> tensor<32x512x64xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 0, 128], sizes = [32, 512, 64], strides = [1, 1, 1] : tensor<32x512x64xf32> -> !flow.dispatch.tensor<readwrite:tensor<32x512x256xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_3 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_3 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_3() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x512x64xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<32x512x256xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [32, 512, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x512x64xf32>> -> tensor<32x512x64xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 0, 192], sizes = [32, 512, 64], strides = [1, 1, 1] : tensor<32x512x64xf32> -> !flow.dispatch.tensor<readwrite:tensor<32x512x256xf32>>
          return
        }
      }
    }
  }
  func.func @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c4194304 = arith.constant 4194304 : index
    %c16777216 = arith.constant 16777216 : index
    %c0 = arith.constant 0 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c64 = arith.constant 64 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c32, %c512, %c64]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<32x512x64xf32> in !stream.resource<external>{%c4194304}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c32, %c512, %c64]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<32x512x64xf32> in !stream.resource<external>{%c4194304}
    hal.buffer_view.assert<%arg2 : !hal.buffer_view> message("input 2") shape([%c32, %c512, %c64]) type(%c553648160_i32) encoding(%c1_i32)
    %2 = stream.tensor.import %arg2 : !hal.buffer_view -> tensor<32x512x64xf32> in !stream.resource<external>{%c4194304}
    hal.buffer_view.assert<%arg3 : !hal.buffer_view> message("input 3") shape([%c32, %c512, %c64]) type(%c553648160_i32) encoding(%c1_i32)
    %3 = stream.tensor.import %arg3 : !hal.buffer_view -> tensor<32x512x64xf32> in !stream.resource<external>{%c4194304}
    %4 = stream.resource.alloc uninitialized : !stream.resource<external>{%c16777216}
    %5 = stream.cmd.execute with(%0 as %arg4: !stream.resource<external>{%c4194304}, %1 as %arg5: !stream.resource<external>{%c4194304}, %2 as %arg6: !stream.resource<external>{%c4194304}, %3 as %arg7: !stream.resource<external>{%c4194304}, %4 as %arg8: !stream.resource<external>{%c16777216}) {
      stream.cmd.dispatch @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_0::@cuda_nvptx_fb::@f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_0 {
        ro %arg4[%c0 for %c4194304] : !stream.resource<external>{%c4194304},
        rw %arg8[%c0 for %c16777216] : !stream.resource<external>{%c16777216}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_1::@cuda_nvptx_fb::@f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_1 {
        ro %arg5[%c0 for %c4194304] : !stream.resource<external>{%c4194304},
        rw %arg8[%c0 for %c16777216] : !stream.resource<external>{%c16777216}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_2::@cuda_nvptx_fb::@f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_2 {
        ro %arg6[%c0 for %c4194304] : !stream.resource<external>{%c4194304},
        rw %arg8[%c0 for %c16777216] : !stream.resource<external>{%c16777216}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_3::@cuda_nvptx_fb::@f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_3 {
        ro %arg7[%c0 for %c4194304] : !stream.resource<external>{%c4194304},
        rw %arg8[%c0 for %c16777216] : !stream.resource<external>{%c16777216}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %6 = stream.timepoint.await %5 => %4 : !stream.resource<external>{%c16777216}
    %7 = stream.tensor.export %6 : tensor<32x512x256xf32> in !stream.resource<external>{%c16777216} -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
}
module @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_0_generic_128x12544x32_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_0_generic_128x12544x32_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x32x12544xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x12544x32xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [128, 32, 12544], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x32x12544xf32>> -> tensor<128x32x12544xf32>
          %3 = tensor.empty() : tensor<128x12544x32xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<128x32x12544xf32>) outs(%3 : tensor<128x12544x32xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<128x12544x32xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [128, 12544, 32], strides = [1, 1, 1] : tensor<128x12544x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x12544x32xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_1_generic_9x32x32_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_1_generic_9x32x32_f32() {
          %c0 = arith.constant 0 : index
          %c205520896 = arith.constant 205520896 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x32x9xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c205520896) : !flow.dispatch.tensor<writeonly:tensor<9x32x32xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [32, 32, 9], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x32x9xf32>> -> tensor<32x32x9xf32>
          %3 = tensor.empty() : tensor<9x32x32xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<32x32x9xf32>) outs(%3 : tensor<9x32x32xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<9x32x32xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [9, 32, 32], strides = [1, 1, 1] : tensor<9x32x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<9x32x32xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_2 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_2 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_2() {
          %c0 = arith.constant 0 : index
          %c205557760 = arith.constant 205557760 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x112x112x32xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c205557760) : !flow.dispatch.tensor<readwrite:tensor<128x114x114x32xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [128, 112, 112, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x112x112x32xf32>> -> tensor<128x112x112x32xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 1, 1, 0], sizes = [128, 112, 112, 32], strides = [1, 1, 1, 1] : tensor<128x112x112x32xf32> -> !flow.dispatch.tensor<readwrite:tensor<128x114x114x32xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_3 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_128x112x112x32x3x3x32_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_128x112x112x32x3x3x32_f32() {
          %c205557760 = arith.constant 205557760 : index
          %c205520896 = arith.constant 205520896 : index
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c205557760) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x114x114x32xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c205520896) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x32x32xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x112x112x32xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [128, 114, 114, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x114x114x32xf32>> -> tensor<128x114x114x32xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 32, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x32x32xf32>> -> tensor<3x3x32x32xf32>
          %5 = tensor.empty() : tensor<128x112x112x32xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x112x112x32xf32>) -> tensor<128x112x112x32xf32>
          %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<128x114x114x32xf32>, tensor<3x3x32x32xf32>) outs(%6 : tensor<128x112x112x32xf32>) -> tensor<128x112x112x32xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [128, 112, 112, 32], strides = [1, 1, 1, 1] : tensor<128x112x112x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x112x112x32xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_4 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_4_generic_128x32x12544_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_4_generic_128x32x12544_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x12544x32xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x32x12544xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [128, 12544, 32], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x12544x32xf32>> -> tensor<128x12544x32xf32>
          %3 = tensor.empty() : tensor<128x32x12544xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<128x12544x32xf32>) outs(%3 : tensor<128x32x12544xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<128x32x12544xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [128, 32, 12544], strides = [1, 1, 1] : tensor<128x32x12544xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x32x12544xf32>>
          return
        }
      }
    }
  }
  func.func @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c418484224 = arith.constant 418484224 : index
    %c205557760 = arith.constant 205557760 : index
    %c205520896 = arith.constant 205520896 : index
    %c36864 = arith.constant 36864 : index
    %c212926464 = arith.constant 212926464 : index
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c112 = arith.constant 112 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c128, %c32, %c112, %c112]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<128x32x112x112xf32> in !stream.resource<external>{%c205520896}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c32, %c32, %c3, %c3]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<32x32x3x3xf32> in !stream.resource<external>{%c36864}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c205520896}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c418484224} => !stream.timepoint
    %3 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c205520896}, %1 as %arg3: !stream.resource<external>{%c36864}, %2 as %arg4: !stream.resource<external>{%c205520896}, %result as %arg5: !stream.resource<transient>{%c418484224}) {
      stream.cmd.concurrent {
        stream.cmd.dispatch @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_0::@cuda_nvptx_fb::@f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_0_generic_128x12544x32_f32 {
          ro %arg2[%c0 for %c205520896] : !stream.resource<external>{%c205520896},
          wo %arg5[%c0 for %c418484224] : !stream.resource<transient>{%c418484224}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.dispatch @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_1::@cuda_nvptx_fb::@f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_1_generic_9x32x32_f32 {
          ro %arg3[%c0 for %c36864] : !stream.resource<external>{%c36864},
          wo %arg5[%c0 for %c418484224] : !stream.resource<transient>{%c418484224}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.fill %c0_i8, %arg5[%c205557760 for %c212926464] : i8 -> !stream.resource<transient>{%c418484224}
      }
      stream.cmd.dispatch @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_2::@cuda_nvptx_fb::@f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_2 {
        ro %arg5[%c0 for %c418484224] : !stream.resource<transient>{%c418484224},
        rw %arg5[%c0 for %c418484224] : !stream.resource<transient>{%c418484224}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_3::@cuda_nvptx_fb::@f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_128x112x112x32x3x3x32_f32 {
        ro %arg5[%c0 for %c418484224] : !stream.resource<transient>{%c418484224},
        wo %arg5[%c0 for %c418484224] : !stream.resource<transient>{%c418484224}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_4::@cuda_nvptx_fb::@f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_4_generic_128x32x12544_f32 {
        ro %arg5[%c0 for %c418484224] : !stream.resource<transient>{%c418484224},
        wo %arg4[%c0 for %c205520896] : !stream.resource<external>{%c205520896}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %4 = stream.resource.dealloca await(%3) => %result : !stream.resource<transient>{%c418484224} => !stream.timepoint
    %5 = stream.timepoint.await %4 => %2 : !stream.resource<external>{%c205520896}
    %6 = stream.tensor.export %5 : tensor<128x32x112x112xf32> in !stream.resource<external>{%c205520896} -> !hal.buffer_view
    return %6 : !hal.buffer_view
  }
}
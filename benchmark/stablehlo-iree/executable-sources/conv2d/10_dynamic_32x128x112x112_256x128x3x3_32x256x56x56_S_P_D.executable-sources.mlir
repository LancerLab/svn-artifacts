module @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_0_generic_32x12544x128_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_0_generic_32x12544x128_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x128x12544xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x12544x128xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [32, 128, 12544], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x128x12544xf32>> -> tensor<32x128x12544xf32>
          %3 = tensor.empty() : tensor<32x12544x128xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<32x128x12544xf32>) outs(%3 : tensor<32x12544x128xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<32x12544x128xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [32, 12544, 128], strides = [1, 1, 1] : tensor<32x12544x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x12544x128xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_1_generic_9x128x256_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_1_generic_9x128x256_f32() {
          %c0 = arith.constant 0 : index
          %c205520896 = arith.constant 205520896 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x128x9xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c205520896) : !flow.dispatch.tensor<writeonly:tensor<9x128x256xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [256, 128, 9], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<256x128x9xf32>> -> tensor<256x128x9xf32>
          %3 = tensor.empty() : tensor<9x128x256xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<256x128x9xf32>) outs(%3 : tensor<9x128x256xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<9x128x256xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [9, 128, 256], strides = [1, 1, 1] : tensor<9x128x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<9x128x256xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_2 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_2 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_2() {
          %c0 = arith.constant 0 : index
          %c206700544 = arith.constant 206700544 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x112x112x128xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c206700544) : !flow.dispatch.tensor<readwrite:tensor<32x114x114x128xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [32, 112, 112, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x112x112x128xf32>> -> tensor<32x112x112x128xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 1, 1, 0], sizes = [32, 112, 112, 128], strides = [1, 1, 1, 1] : tensor<32x112x112x128xf32> -> !flow.dispatch.tensor<readwrite:tensor<32x114x114x128xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_3 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_32x56x56x256x3x3x128_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_32x56x56x256x3x3x128_f32() {
          %c206700544 = arith.constant 206700544 : index
          %c205520896 = arith.constant 205520896 : index
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c206700544) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x114x114x128xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c205520896) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x128x256xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x56x56x256xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [32, 114, 114, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x114x114x128xf32>> -> tensor<32x114x114x128xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 128, 256], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x128x256xf32>> -> tensor<3x3x128x256xf32>
          %5 = tensor.empty() : tensor<32x56x56x256xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<32x56x56x256xf32>) -> tensor<32x56x56x256xf32>
          %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<32x114x114x128xf32>, tensor<3x3x128x256xf32>) outs(%6 : tensor<32x56x56x256xf32>) -> tensor<32x56x56x256xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [32, 56, 56, 256], strides = [1, 1, 1, 1] : tensor<32x56x56x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x56x56x256xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_4 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_4_generic_32x256x3136_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_4_generic_32x256x3136_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x3136x256xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x256x3136xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [32, 3136, 256], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x3136x256xf32>> -> tensor<32x3136x256xf32>
          %3 = tensor.empty() : tensor<32x256x3136xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<32x3136x256xf32>) outs(%3 : tensor<32x256x3136xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<32x256x3136xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [32, 256, 3136], strides = [1, 1, 1] : tensor<32x256x3136xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x256x3136xf32>>
          return
        }
      }
    }
  }
  func.func @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c419627008 = arith.constant 419627008 : index
    %c206700544 = arith.constant 206700544 : index
    %c205520896 = arith.constant 205520896 : index
    %c1179648 = arith.constant 1179648 : index
    %c212926464 = arith.constant 212926464 : index
    %c102760448 = arith.constant 102760448 : index
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c256 = arith.constant 256 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c112 = arith.constant 112 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c32, %c128, %c112, %c112]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<32x128x112x112xf32> in !stream.resource<external>{%c205520896}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c256, %c128, %c3, %c3]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<256x128x3x3xf32> in !stream.resource<external>{%c1179648}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c102760448}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c419627008} => !stream.timepoint
    %3 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c205520896}, %1 as %arg3: !stream.resource<external>{%c1179648}, %2 as %arg4: !stream.resource<external>{%c102760448}, %result as %arg5: !stream.resource<transient>{%c419627008}) {
      stream.cmd.concurrent {
        stream.cmd.dispatch @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_0::@cuda_nvptx_fb::@f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_0_generic_32x12544x128_f32 {
          ro %arg2[%c0 for %c205520896] : !stream.resource<external>{%c205520896},
          wo %arg5[%c0 for %c419627008] : !stream.resource<transient>{%c419627008}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.dispatch @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_1::@cuda_nvptx_fb::@f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_1_generic_9x128x256_f32 {
          ro %arg3[%c0 for %c1179648] : !stream.resource<external>{%c1179648},
          wo %arg5[%c0 for %c419627008] : !stream.resource<transient>{%c419627008}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.fill %c0_i8, %arg5[%c206700544 for %c212926464] : i8 -> !stream.resource<transient>{%c419627008}
      }
      stream.cmd.dispatch @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_2::@cuda_nvptx_fb::@f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_2 {
        ro %arg5[%c0 for %c419627008] : !stream.resource<transient>{%c419627008},
        rw %arg5[%c0 for %c419627008] : !stream.resource<transient>{%c419627008}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_3::@cuda_nvptx_fb::@f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_32x56x56x256x3x3x128_f32 {
        ro %arg5[%c0 for %c419627008] : !stream.resource<transient>{%c419627008},
        wo %arg5[%c0 for %c419627008] : !stream.resource<transient>{%c419627008}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_4::@cuda_nvptx_fb::@f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_dispatch_4_generic_32x256x3136_f32 {
        ro %arg5[%c0 for %c419627008] : !stream.resource<transient>{%c419627008},
        wo %arg4[%c0 for %c102760448] : !stream.resource<external>{%c102760448}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %4 = stream.resource.dealloca await(%3) => %result : !stream.resource<transient>{%c419627008} => !stream.timepoint
    %5 = stream.timepoint.await %4 => %2 : !stream.resource<external>{%c102760448}
    %6 = stream.tensor.export %5 : tensor<32x256x56x56xf32> in !stream.resource<external>{%c102760448} -> !hal.buffer_view
    return %6 : !hal.buffer_view
  }
}
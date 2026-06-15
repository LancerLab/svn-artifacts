module @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_0_generic_16x16384x64_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_0_generic_16x16384x64_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x64x16384xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x16384x64xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 64, 16384], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x64x16384xf32>> -> tensor<16x64x16384xf32>
          %3 = tensor.empty() : tensor<16x16384x64xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<16x64x16384xf32>) outs(%3 : tensor<16x16384x64xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<16x16384x64xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [16, 16384, 64], strides = [1, 1, 1] : tensor<16x16384x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x16384x64xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_1_generic_9x64x128_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_1_generic_9x64x128_f32() {
          %c0 = arith.constant 0 : index
          %c67108864 = arith.constant 67108864 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x64x9xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c67108864) : !flow.dispatch.tensor<writeonly:tensor<9x64x128xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [128, 64, 9], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x64x9xf32>> -> tensor<128x64x9xf32>
          %3 = tensor.empty() : tensor<9x64x128xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<128x64x9xf32>) outs(%3 : tensor<9x64x128xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<9x64x128xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [9, 64, 128], strides = [1, 1, 1] : tensor<9x64x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<9x64x128xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_2 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_2 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_2() {
          %c0 = arith.constant 0 : index
          %c67403776 = arith.constant 67403776 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x128x128x64xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c67403776) : !flow.dispatch.tensor<readwrite:tensor<16x130x130x64xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [16, 128, 128, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x128x128x64xf32>> -> tensor<16x128x128x64xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 1, 1, 0], sizes = [16, 128, 128, 64], strides = [1, 1, 1, 1] : tensor<16x128x128x64xf32> -> !flow.dispatch.tensor<readwrite:tensor<16x130x130x64xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_3 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_16x128x128x128x3x3x64_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_16x128x128x128x3x3x64_f32() {
          %c67403776 = arith.constant 67403776 : index
          %c67108864 = arith.constant 67108864 : index
          %c136626176 = arith.constant 136626176 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c67403776) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x130x130x64xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c67108864) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x64x128xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c136626176) : !flow.dispatch.tensor<writeonly:tensor<16x128x128x128xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [16, 130, 130, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x130x130x64xf32>> -> tensor<16x130x130x64xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 64, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x64x128xf32>> -> tensor<3x3x64x128xf32>
          %5 = tensor.empty() : tensor<16x128x128x128xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<16x128x128x128xf32>) -> tensor<16x128x128x128xf32>
          %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<16x130x130x64xf32>, tensor<3x3x64x128xf32>) outs(%6 : tensor<16x128x128x128xf32>) -> tensor<16x128x128x128xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [16, 128, 128, 128], strides = [1, 1, 1, 1] : tensor<16x128x128x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x128x128x128xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_4 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_4_generic_16x128x16384_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_4_generic_16x128x16384_f32() {
          %c136626176 = arith.constant 136626176 : index
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c136626176) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x16384x128xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x128x16384xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 16384, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x16384x128xf32>> -> tensor<16x16384x128xf32>
          %3 = tensor.empty() : tensor<16x128x16384xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<16x16384x128xf32>) outs(%3 : tensor<16x128x16384xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<16x128x16384xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [16, 128, 16384], strides = [1, 1, 1] : tensor<16x128x16384xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x128x16384xf32>>
          return
        }
      }
    }
  }
  func.func @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c270843904 = arith.constant 270843904 : index
    %c67403776 = arith.constant 67403776 : index
    %c67108864 = arith.constant 67108864 : index
    %c294912 = arith.constant 294912 : index
    %c69222400 = arith.constant 69222400 : index
    %c134217728 = arith.constant 134217728 : index
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c16, %c64, %c128, %c128]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<16x64x128x128xf32> in !stream.resource<external>{%c67108864}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c128, %c64, %c3, %c3]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<128x64x3x3xf32> in !stream.resource<external>{%c294912}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c134217728}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c270843904} => !stream.timepoint
    %3 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c67108864}, %1 as %arg3: !stream.resource<external>{%c294912}, %2 as %arg4: !stream.resource<external>{%c134217728}, %result as %arg5: !stream.resource<transient>{%c270843904}) {
      stream.cmd.concurrent {
        stream.cmd.dispatch @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_0::@cuda_nvptx_fb::@f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_0_generic_16x16384x64_f32 {
          ro %arg2[%c0 for %c67108864] : !stream.resource<external>{%c67108864},
          wo %arg5[%c0 for %c270843904] : !stream.resource<transient>{%c270843904}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.dispatch @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_1::@cuda_nvptx_fb::@f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_1_generic_9x64x128_f32 {
          ro %arg3[%c0 for %c294912] : !stream.resource<external>{%c294912},
          wo %arg5[%c0 for %c270843904] : !stream.resource<transient>{%c270843904}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.fill %c0_i8, %arg5[%c67403776 for %c69222400] : i8 -> !stream.resource<transient>{%c270843904}
      }
      stream.cmd.dispatch @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_2::@cuda_nvptx_fb::@f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_2 {
        ro %arg5[%c0 for %c270843904] : !stream.resource<transient>{%c270843904},
        rw %arg5[%c0 for %c270843904] : !stream.resource<transient>{%c270843904}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_3::@cuda_nvptx_fb::@f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_16x128x128x128x3x3x64_f32 {
        ro %arg5[%c0 for %c270843904] : !stream.resource<transient>{%c270843904},
        wo %arg5[%c0 for %c270843904] : !stream.resource<transient>{%c270843904}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_4::@cuda_nvptx_fb::@f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D_dispatch_4_generic_16x128x16384_f32 {
        ro %arg5[%c0 for %c270843904] : !stream.resource<transient>{%c270843904},
        wo %arg4[%c0 for %c134217728] : !stream.resource<external>{%c134217728}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %4 = stream.resource.dealloca await(%3) => %result : !stream.resource<transient>{%c270843904} => !stream.timepoint
    %5 = stream.timepoint.await %4 => %2 : !stream.resource<external>{%c134217728}
    %6 = stream.tensor.export %5 : tensor<16x128x128x128xf32> in !stream.resource<external>{%c134217728} -> !hal.buffer_view
    return %6 : !hal.buffer_view
  }
}
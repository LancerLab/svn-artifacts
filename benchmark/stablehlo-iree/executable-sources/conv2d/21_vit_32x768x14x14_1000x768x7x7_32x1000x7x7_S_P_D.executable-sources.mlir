module @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_0_generic_32x196x768_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_0_generic_32x196x768_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x768x196xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x196x768xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [32, 768, 196], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x768x196xf32>> -> tensor<32x768x196xf32>
          %3 = tensor.empty() : tensor<32x196x768xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<32x768x196xf32>) outs(%3 : tensor<32x196x768xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<32x196x768xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [32, 196, 768], strides = [1, 1, 1] : tensor<32x196x768xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x196x768xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_1_generic_49x768x1000_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_1_generic_49x768x1000_f32() {
          %c0 = arith.constant 0 : index
          %c19267584 = arith.constant 19267584 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1000x768x49xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c19267584) : !flow.dispatch.tensor<writeonly:tensor<49x768x1000xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1000, 768, 49], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1000x768x49xf32>> -> tensor<1000x768x49xf32>
          %3 = tensor.empty() : tensor<49x768x1000xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1000x768x49xf32>) outs(%3 : tensor<49x768x1000xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<49x768x1000xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [49, 768, 1000], strides = [1, 1, 1] : tensor<49x768x1000xf32> -> !flow.dispatch.tensor<writeonly:tensor<49x768x1000xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_2 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_2 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_2() {
          %c0 = arith.constant 0 : index
          %c169795584 = arith.constant 169795584 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x14x14x768xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c169795584) : !flow.dispatch.tensor<readwrite:tensor<32x20x20x768xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [32, 14, 14, 768], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x14x14x768xf32>> -> tensor<32x14x14x768xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 3, 3, 0], sizes = [32, 14, 14, 768], strides = [1, 1, 1, 1] : tensor<32x14x14x768xf32> -> !flow.dispatch.tensor<readwrite:tensor<32x20x20x768xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_3 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_32x7x7x1000x7x7x768_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_32x7x7x1000x7x7x768_f32() {
          %c169795584 = arith.constant 169795584 : index
          %c19267584 = arith.constant 19267584 : index
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c169795584) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x20x20x768xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c19267584) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<7x7x768x1000xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x7x7x1000xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [32, 20, 20, 768], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x20x20x768xf32>> -> tensor<32x20x20x768xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [7, 7, 768, 1000], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<7x7x768x1000xf32>> -> tensor<7x7x768x1000xf32>
          %5 = tensor.empty() : tensor<32x7x7x1000xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<32x7x7x1000xf32>) -> tensor<32x7x7x1000xf32>
          %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<32x20x20x768xf32>, tensor<7x7x768x1000xf32>) outs(%6 : tensor<32x7x7x1000xf32>) -> tensor<32x7x7x1000xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [32, 7, 7, 1000], strides = [1, 1, 1, 1] : tensor<32x7x7x1000xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x7x7x1000xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_4 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_4_generic_32x1000x49_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_4_generic_32x1000x49_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x49x1000xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x1000x49xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [32, 49, 1000], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x49x1000xf32>> -> tensor<32x49x1000xf32>
          %3 = tensor.empty() : tensor<32x1000x49xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<32x49x1000xf32>) outs(%3 : tensor<32x1000x49xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<32x1000x49xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [32, 1000, 49], strides = [1, 1, 1] : tensor<32x1000x49xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x1000x49xf32>>
          return
        }
      }
    }
  }
  func.func @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c209117184 = arith.constant 209117184 : index
    %c169795584 = arith.constant 169795584 : index
    %c19267584 = arith.constant 19267584 : index
    %c150528000 = arith.constant 150528000 : index
    %c39321600 = arith.constant 39321600 : index
    %c6272000 = arith.constant 6272000 : index
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c1000 = arith.constant 1000 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32 = arith.constant 32 : index
    %c768 = arith.constant 768 : index
    %c14 = arith.constant 14 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c32, %c768, %c14, %c14]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<32x768x14x14xf32> in !stream.resource<external>{%c19267584}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c1000, %c768, %c7, %c7]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<1000x768x7x7xf32> in !stream.resource<external>{%c150528000}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c6272000}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c209117184} => !stream.timepoint
    %3 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c19267584}, %1 as %arg3: !stream.resource<external>{%c150528000}, %2 as %arg4: !stream.resource<external>{%c6272000}, %result as %arg5: !stream.resource<transient>{%c209117184}) {
      stream.cmd.concurrent {
        stream.cmd.dispatch @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_0::@cuda_nvptx_fb::@f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_0_generic_32x196x768_f32 {
          ro %arg2[%c0 for %c19267584] : !stream.resource<external>{%c19267584},
          wo %arg5[%c0 for %c209117184] : !stream.resource<transient>{%c209117184}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.dispatch @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_1::@cuda_nvptx_fb::@f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_1_generic_49x768x1000_f32 {
          ro %arg3[%c0 for %c150528000] : !stream.resource<external>{%c150528000},
          wo %arg5[%c0 for %c209117184] : !stream.resource<transient>{%c209117184}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.fill %c0_i8, %arg5[%c169795584 for %c39321600] : i8 -> !stream.resource<transient>{%c209117184}
      }
      stream.cmd.dispatch @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_2::@cuda_nvptx_fb::@f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_2 {
        ro %arg5[%c0 for %c209117184] : !stream.resource<transient>{%c209117184},
        rw %arg5[%c0 for %c209117184] : !stream.resource<transient>{%c209117184}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_3::@cuda_nvptx_fb::@f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_32x7x7x1000x7x7x768_f32 {
        ro %arg5[%c0 for %c209117184] : !stream.resource<transient>{%c209117184},
        wo %arg5[%c0 for %c209117184] : !stream.resource<transient>{%c209117184}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_4::@cuda_nvptx_fb::@f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_dispatch_4_generic_32x1000x49_f32 {
        ro %arg5[%c0 for %c209117184] : !stream.resource<transient>{%c209117184},
        wo %arg4[%c0 for %c6272000] : !stream.resource<external>{%c6272000}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %4 = stream.resource.dealloca await(%3) => %result : !stream.resource<transient>{%c209117184} => !stream.timepoint
    %5 = stream.timepoint.await %4 => %2 : !stream.resource<external>{%c6272000}
    %6 = stream.tensor.export %5 : tensor<32x1000x7x7xf32> in !stream.resource<external>{%c6272000} -> !hal.buffer_view
    return %6 : !hal.buffer_view
  }
}
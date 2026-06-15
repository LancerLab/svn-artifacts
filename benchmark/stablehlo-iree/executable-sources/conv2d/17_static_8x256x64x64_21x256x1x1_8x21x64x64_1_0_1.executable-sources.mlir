module @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_dispatch_0_generic_8x4096x256_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_dispatch_0_generic_8x4096x256_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x256x4096xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x4096x256xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [8, 256, 4096], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<8x256x4096xf32>> -> tensor<8x256x4096xf32>
          %3 = tensor.empty() : tensor<8x4096x256xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<8x256x4096xf32>) outs(%3 : tensor<8x4096x256xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<8x4096x256xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [8, 4096, 256], strides = [1, 1, 1] : tensor<8x4096x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<8x4096x256xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_dispatch_1_generic_256x21_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_dispatch_1_generic_256x21_f32() {
          %c0 = arith.constant 0 : index
          %c33554432 = arith.constant 33554432 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<21x256xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c33554432) : !flow.dispatch.tensor<writeonly:tensor<256x21xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [21, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<21x256xf32>> -> tensor<21x256xf32>
          %3 = tensor.empty() : tensor<256x21xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<21x256xf32>) outs(%3 : tensor<256x21xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<256x21xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [256, 21], strides = [1, 1] : tensor<256x21xf32> -> !flow.dispatch.tensor<writeonly:tensor<256x21xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_dispatch_2 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_8x64x64x21x1x1x256_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_8x64x64x21x1x1x256_f32() {
          %c0 = arith.constant 0 : index
          %c33554432 = arith.constant 33554432 : index
          %c33575936 = arith.constant 33575936 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x64x64x256xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c33554432) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1x256x21xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c33575936) : !flow.dispatch.tensor<writeonly:tensor<8x64x64x21xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [8, 64, 64, 256], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<8x64x64x256xf32>> -> tensor<8x64x64x256xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [1, 1, 256, 21], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x256x21xf32>> -> tensor<1x1x256x21xf32>
          %5 = tensor.empty() : tensor<8x64x64x21xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<8x64x64x21xf32>) -> tensor<8x64x64x21xf32>
          %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<8x64x64x256xf32>, tensor<1x1x256x21xf32>) outs(%6 : tensor<8x64x64x21xf32>) -> tensor<8x64x64x21xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [8, 64, 64, 21], strides = [1, 1, 1, 1] : tensor<8x64x64x21xf32> -> !flow.dispatch.tensor<writeonly:tensor<8x64x64x21xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_dispatch_3 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_dispatch_3_generic_8x21x4096_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_dispatch_3_generic_8x21x4096_f32() {
          %c33575936 = arith.constant 33575936 : index
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c33575936) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x4096x21xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x21x4096xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [8, 4096, 21], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<8x4096x21xf32>> -> tensor<8x4096x21xf32>
          %3 = tensor.empty() : tensor<8x21x4096xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<8x4096x21xf32>) outs(%3 : tensor<8x21x4096xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<8x21x4096xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [8, 21, 4096], strides = [1, 1, 1] : tensor<8x21x4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<8x21x4096xf32>>
          return
        }
      }
    }
  }
  func.func @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c36328448 = arith.constant 36328448 : index
    %c33554432 = arith.constant 33554432 : index
    %c21504 = arith.constant 21504 : index
    %c2752512 = arith.constant 2752512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c21 = arith.constant 21 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c8, %c256, %c64, %c64]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<8x256x64x64xf32> in !stream.resource<external>{%c33554432}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c21, %c256, %c1, %c1]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<21x256x1x1xf32> in !stream.resource<external>{%c21504}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c2752512}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c36328448} => !stream.timepoint
    %3 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c33554432}, %1 as %arg3: !stream.resource<external>{%c21504}, %2 as %arg4: !stream.resource<external>{%c2752512}, %result as %arg5: !stream.resource<transient>{%c36328448}) {
      stream.cmd.concurrent {
        stream.cmd.dispatch @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_dispatch_0::@cuda_nvptx_fb::@f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_dispatch_0_generic_8x4096x256_f32 {
          ro %arg2[%c0 for %c33554432] : !stream.resource<external>{%c33554432},
          wo %arg5[%c0 for %c36328448] : !stream.resource<transient>{%c36328448}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.dispatch @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_dispatch_1::@cuda_nvptx_fb::@f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_dispatch_1_generic_256x21_f32 {
          ro %arg3[%c0 for %c21504] : !stream.resource<external>{%c21504},
          wo %arg5[%c0 for %c36328448] : !stream.resource<transient>{%c36328448}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      }
      stream.cmd.dispatch @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_dispatch_2::@cuda_nvptx_fb::@f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_8x64x64x21x1x1x256_f32 {
        ro %arg5[%c0 for %c36328448] : !stream.resource<transient>{%c36328448},
        wo %arg5[%c0 for %c36328448] : !stream.resource<transient>{%c36328448}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_dispatch_3::@cuda_nvptx_fb::@f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_dispatch_3_generic_8x21x4096_f32 {
        ro %arg5[%c0 for %c36328448] : !stream.resource<transient>{%c36328448},
        wo %arg4[%c0 for %c2752512] : !stream.resource<external>{%c2752512}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %4 = stream.resource.dealloca await(%3) => %result : !stream.resource<transient>{%c36328448} => !stream.timepoint
    %5 = stream.timepoint.await %4 => %2 : !stream.resource<external>{%c2752512}
    %6 = stream.tensor.export %5 : tensor<8x21x64x64xf32> in !stream.resource<external>{%c2752512} -> !hal.buffer_view
    return %6 : !hal.buffer_view
  }
}
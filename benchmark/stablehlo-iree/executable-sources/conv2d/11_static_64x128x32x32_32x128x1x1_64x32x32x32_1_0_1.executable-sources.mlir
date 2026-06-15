module @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_0_generic_64x1024x128_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_0_generic_64x1024x128_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x128x1024xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x1024x128xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 128, 1024], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x128x1024xf32>> -> tensor<64x128x1024xf32>
          %3 = tensor.empty() : tensor<64x1024x128xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<64x128x1024xf32>) outs(%3 : tensor<64x1024x128xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<64x1024x128xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [64, 1024, 128], strides = [1, 1, 1] : tensor<64x1024x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x1024x128xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_1_generic_128x32_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_1_generic_128x32_f32() {
          %c0 = arith.constant 0 : index
          %c33554432 = arith.constant 33554432 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x128xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c33554432) : !flow.dispatch.tensor<writeonly:tensor<128x32xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x128xf32>> -> tensor<32x128xf32>
          %3 = tensor.empty() : tensor<128x32xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<32x128xf32>) outs(%3 : tensor<128x32xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<128x32xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [128, 32], strides = [1, 1] : tensor<128x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x32xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_2 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_64x32x32x32x1x1x128_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_64x32x32x32x1x1x128_f32() {
          %c0 = arith.constant 0 : index
          %c33554432 = arith.constant 33554432 : index
          %c33570816 = arith.constant 33570816 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x32x32x128xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c33554432) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1x128x32xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c33570816) : !flow.dispatch.tensor<writeonly:tensor<64x32x32x32xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [64, 32, 32, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x32x32x128xf32>> -> tensor<64x32x32x128xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [1, 1, 128, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x128x32xf32>> -> tensor<1x1x128x32xf32>
          %5 = tensor.empty() : tensor<64x32x32x32xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<64x32x32x32xf32>) -> tensor<64x32x32x32xf32>
          %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<64x32x32x128xf32>, tensor<1x1x128x32xf32>) outs(%6 : tensor<64x32x32x32xf32>) -> tensor<64x32x32x32xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [64, 32, 32, 32], strides = [1, 1, 1, 1] : tensor<64x32x32x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x32x32x32xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_3 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_3_generic_64x32x1024_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_3_generic_64x32x1024_f32() {
          %c33570816 = arith.constant 33570816 : index
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c33570816) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x1024x32xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x32x1024xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 1024, 32], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x1024x32xf32>> -> tensor<64x1024x32xf32>
          %3 = tensor.empty() : tensor<64x32x1024xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<64x1024x32xf32>) outs(%3 : tensor<64x32x1024xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<64x32x1024xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [64, 32, 1024], strides = [1, 1, 1] : tensor<64x32x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x32x1024xf32>>
          return
        }
      }
    }
  }
  func.func @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c41959424 = arith.constant 41959424 : index
    %c33554432 = arith.constant 33554432 : index
    %c16384 = arith.constant 16384 : index
    %c8388608 = arith.constant 8388608 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c64, %c128, %c32, %c32]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<64x128x32x32xf32> in !stream.resource<external>{%c33554432}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c32, %c128, %c1, %c1]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<32x128x1x1xf32> in !stream.resource<external>{%c16384}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c8388608}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c41959424} => !stream.timepoint
    %3 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c33554432}, %1 as %arg3: !stream.resource<external>{%c16384}, %2 as %arg4: !stream.resource<external>{%c8388608}, %result as %arg5: !stream.resource<transient>{%c41959424}) {
      stream.cmd.concurrent {
        stream.cmd.dispatch @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_0::@cuda_nvptx_fb::@f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_0_generic_64x1024x128_f32 {
          ro %arg2[%c0 for %c33554432] : !stream.resource<external>{%c33554432},
          wo %arg5[%c0 for %c41959424] : !stream.resource<transient>{%c41959424}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.dispatch @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_1::@cuda_nvptx_fb::@f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_1_generic_128x32_f32 {
          ro %arg3[%c0 for %c16384] : !stream.resource<external>{%c16384},
          wo %arg5[%c0 for %c41959424] : !stream.resource<transient>{%c41959424}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      }
      stream.cmd.dispatch @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_2::@cuda_nvptx_fb::@f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_64x32x32x32x1x1x128_f32 {
        ro %arg5[%c0 for %c41959424] : !stream.resource<transient>{%c41959424},
        wo %arg5[%c0 for %c41959424] : !stream.resource<transient>{%c41959424}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_3::@cuda_nvptx_fb::@f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_3_generic_64x32x1024_f32 {
        ro %arg5[%c0 for %c41959424] : !stream.resource<transient>{%c41959424},
        wo %arg4[%c0 for %c8388608] : !stream.resource<external>{%c8388608}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %4 = stream.resource.dealloca await(%3) => %result : !stream.resource<transient>{%c41959424} => !stream.timepoint
    %5 = stream.timepoint.await %4 => %2 : !stream.resource<external>{%c8388608}
    %6 = stream.tensor.export %5 : tensor<64x32x32x32xf32> in !stream.resource<external>{%c8388608} -> !hal.buffer_view
    return %6 : !hal.buffer_view
  }
}
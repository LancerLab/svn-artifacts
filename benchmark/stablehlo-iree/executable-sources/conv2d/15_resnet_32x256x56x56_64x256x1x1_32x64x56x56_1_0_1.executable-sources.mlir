module @f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1_dispatch_0_generic_32x3136x256_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1_dispatch_0_generic_32x3136x256_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x256x3136xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x3136x256xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [32, 256, 3136], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x256x3136xf32>> -> tensor<32x256x3136xf32>
          %3 = tensor.empty() : tensor<32x3136x256xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<32x256x3136xf32>) outs(%3 : tensor<32x3136x256xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<32x3136x256xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [32, 3136, 256], strides = [1, 1, 1] : tensor<32x3136x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x3136x256xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1_dispatch_1_generic_256x64_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1_dispatch_1_generic_256x64_f32() {
          %c0 = arith.constant 0 : index
          %c102760448 = arith.constant 102760448 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x256xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c102760448) : !flow.dispatch.tensor<writeonly:tensor<256x64xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x256xf32>> -> tensor<64x256xf32>
          %3 = tensor.empty() : tensor<256x64xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<64x256xf32>) outs(%3 : tensor<256x64xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<256x64xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [256, 64], strides = [1, 1] : tensor<256x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<256x64xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1_dispatch_2 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_32x56x56x64x1x1x256_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_32x56x56x64x1x1x256_f32() {
          %c0 = arith.constant 0 : index
          %c102760448 = arith.constant 102760448 : index
          %c102825984 = arith.constant 102825984 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x56x56x256xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c102760448) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1x256x64xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c102825984) : !flow.dispatch.tensor<writeonly:tensor<32x56x56x64xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [32, 56, 56, 256], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x56x56x256xf32>> -> tensor<32x56x56x256xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [1, 1, 256, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x256x64xf32>> -> tensor<1x1x256x64xf32>
          %5 = tensor.empty() : tensor<32x56x56x64xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<32x56x56x64xf32>) -> tensor<32x56x56x64xf32>
          %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<32x56x56x256xf32>, tensor<1x1x256x64xf32>) outs(%6 : tensor<32x56x56x64xf32>) -> tensor<32x56x56x64xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [32, 56, 56, 64], strides = [1, 1, 1, 1] : tensor<32x56x56x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x56x56x64xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1_dispatch_3 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1_dispatch_3_generic_32x64x3136_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1_dispatch_3_generic_32x64x3136_f32() {
          %c102825984 = arith.constant 102825984 : index
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c102825984) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x3136x64xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x64x3136xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [32, 3136, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x3136x64xf32>> -> tensor<32x3136x64xf32>
          %3 = tensor.empty() : tensor<32x64x3136xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<32x3136x64xf32>) outs(%3 : tensor<32x64x3136xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<32x64x3136xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [32, 64, 3136], strides = [1, 1, 1] : tensor<32x64x3136xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x64x3136xf32>>
          return
        }
      }
    }
  }
  func.func @f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c128516096 = arith.constant 128516096 : index
    %c102760448 = arith.constant 102760448 : index
    %c65536 = arith.constant 65536 : index
    %c25690112 = arith.constant 25690112 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32 = arith.constant 32 : index
    %c256 = arith.constant 256 : index
    %c56 = arith.constant 56 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c32, %c256, %c56, %c56]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<32x256x56x56xf32> in !stream.resource<external>{%c102760448}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c64, %c256, %c1, %c1]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<64x256x1x1xf32> in !stream.resource<external>{%c65536}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c25690112}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c128516096} => !stream.timepoint
    %3 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c102760448}, %1 as %arg3: !stream.resource<external>{%c65536}, %2 as %arg4: !stream.resource<external>{%c25690112}, %result as %arg5: !stream.resource<transient>{%c128516096}) {
      stream.cmd.concurrent {
        stream.cmd.dispatch @f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1_dispatch_0::@cuda_nvptx_fb::@f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1_dispatch_0_generic_32x3136x256_f32 {
          ro %arg2[%c0 for %c102760448] : !stream.resource<external>{%c102760448},
          wo %arg5[%c0 for %c128516096] : !stream.resource<transient>{%c128516096}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.dispatch @f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1_dispatch_1::@cuda_nvptx_fb::@f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1_dispatch_1_generic_256x64_f32 {
          ro %arg3[%c0 for %c65536] : !stream.resource<external>{%c65536},
          wo %arg5[%c0 for %c128516096] : !stream.resource<transient>{%c128516096}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      }
      stream.cmd.dispatch @f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1_dispatch_2::@cuda_nvptx_fb::@f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_32x56x56x64x1x1x256_f32 {
        ro %arg5[%c0 for %c128516096] : !stream.resource<transient>{%c128516096},
        wo %arg5[%c0 for %c128516096] : !stream.resource<transient>{%c128516096}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1_dispatch_3::@cuda_nvptx_fb::@f_15_resnet_32x256x56x56_64x256x1x1_32x64x56x56_1_0_1_dispatch_3_generic_32x64x3136_f32 {
        ro %arg5[%c0 for %c128516096] : !stream.resource<transient>{%c128516096},
        wo %arg4[%c0 for %c25690112] : !stream.resource<external>{%c25690112}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %4 = stream.resource.dealloca await(%3) => %result : !stream.resource<transient>{%c128516096} => !stream.timepoint
    %5 = stream.timepoint.await %4 => %2 : !stream.resource<external>{%c25690112}
    %6 = stream.tensor.export %5 : tensor<32x64x56x56xf32> in !stream.resource<external>{%c25690112} -> !hal.buffer_view
    return %6 : !hal.buffer_view
  }
}
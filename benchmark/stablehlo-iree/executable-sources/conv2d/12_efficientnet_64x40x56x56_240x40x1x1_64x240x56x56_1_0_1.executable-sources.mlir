module @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_0_generic_64x3136x40_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_0_generic_64x3136x40_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x40x3136xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x3136x40xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 40, 3136], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x40x3136xf32>> -> tensor<64x40x3136xf32>
          %3 = tensor.empty() : tensor<64x3136x40xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<64x40x3136xf32>) outs(%3 : tensor<64x3136x40xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<64x3136x40xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [64, 3136, 40], strides = [1, 1, 1] : tensor<64x3136x40xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x3136x40xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_1_generic_40x240_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_1_generic_40x240_f32() {
          %c0 = arith.constant 0 : index
          %c32112640 = arith.constant 32112640 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<240x40xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c32112640) : !flow.dispatch.tensor<writeonly:tensor<40x240xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [240, 40], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<240x40xf32>> -> tensor<240x40xf32>
          %3 = tensor.empty() : tensor<40x240xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<240x40xf32>) outs(%3 : tensor<40x240xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<40x240xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [40, 240], strides = [1, 1] : tensor<40x240xf32> -> !flow.dispatch.tensor<writeonly:tensor<40x240xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_2 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_64x56x56x240x1x1x40_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_64x56x56x240x1x1x40_f32() {
          %c0 = arith.constant 0 : index
          %c32112640 = arith.constant 32112640 : index
          %c32151040 = arith.constant 32151040 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x56x56x40xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c32112640) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1x40x240xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c32151040) : !flow.dispatch.tensor<writeonly:tensor<64x56x56x240xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [64, 56, 56, 40], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x56x56x40xf32>> -> tensor<64x56x56x40xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [1, 1, 40, 240], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x40x240xf32>> -> tensor<1x1x40x240xf32>
          %5 = tensor.empty() : tensor<64x56x56x240xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<64x56x56x240xf32>) -> tensor<64x56x56x240xf32>
          %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<64x56x56x40xf32>, tensor<1x1x40x240xf32>) outs(%6 : tensor<64x56x56x240xf32>) -> tensor<64x56x56x240xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [64, 56, 56, 240], strides = [1, 1, 1, 1] : tensor<64x56x56x240xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x56x56x240xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_3 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_3_generic_64x240x3136_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_3_generic_64x240x3136_f32() {
          %c32151040 = arith.constant 32151040 : index
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c32151040) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x3136x240xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x240x3136xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 3136, 240], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x3136x240xf32>> -> tensor<64x3136x240xf32>
          %3 = tensor.empty() : tensor<64x240x3136xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<64x3136x240xf32>) outs(%3 : tensor<64x240x3136xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<64x240x3136xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [64, 240, 3136], strides = [1, 1, 1] : tensor<64x240x3136xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x240x3136xf32>>
          return
        }
      }
    }
  }
  func.func @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c224826880 = arith.constant 224826880 : index
    %c32112640 = arith.constant 32112640 : index
    %c38400 = arith.constant 38400 : index
    %c192675840 = arith.constant 192675840 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c240 = arith.constant 240 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c40 = arith.constant 40 : index
    %c56 = arith.constant 56 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c64, %c40, %c56, %c56]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<64x40x56x56xf32> in !stream.resource<external>{%c32112640}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c240, %c40, %c1, %c1]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<240x40x1x1xf32> in !stream.resource<external>{%c38400}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c192675840}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c224826880} => !stream.timepoint
    %3 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c32112640}, %1 as %arg3: !stream.resource<external>{%c38400}, %2 as %arg4: !stream.resource<external>{%c192675840}, %result as %arg5: !stream.resource<transient>{%c224826880}) {
      stream.cmd.concurrent {
        stream.cmd.dispatch @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_0::@cuda_nvptx_fb::@f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_0_generic_64x3136x40_f32 {
          ro %arg2[%c0 for %c32112640] : !stream.resource<external>{%c32112640},
          wo %arg5[%c0 for %c224826880] : !stream.resource<transient>{%c224826880}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.dispatch @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_1::@cuda_nvptx_fb::@f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_1_generic_40x240_f32 {
          ro %arg3[%c0 for %c38400] : !stream.resource<external>{%c38400},
          wo %arg5[%c0 for %c224826880] : !stream.resource<transient>{%c224826880}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      }
      stream.cmd.dispatch @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_2::@cuda_nvptx_fb::@f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_64x56x56x240x1x1x40_f32 {
        ro %arg5[%c0 for %c224826880] : !stream.resource<transient>{%c224826880},
        wo %arg5[%c0 for %c224826880] : !stream.resource<transient>{%c224826880}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_3::@cuda_nvptx_fb::@f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_3_generic_64x240x3136_f32 {
        ro %arg5[%c0 for %c224826880] : !stream.resource<transient>{%c224826880},
        wo %arg4[%c0 for %c192675840] : !stream.resource<external>{%c192675840}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %4 = stream.resource.dealloca await(%3) => %result : !stream.resource<transient>{%c224826880} => !stream.timepoint
    %5 = stream.timepoint.await %4 => %2 : !stream.resource<external>{%c192675840}
    %6 = stream.tensor.export %5 : tensor<64x240x56x56xf32> in !stream.resource<external>{%c192675840} -> !hal.buffer_view
    return %6 : !hal.buffer_view
  }
}
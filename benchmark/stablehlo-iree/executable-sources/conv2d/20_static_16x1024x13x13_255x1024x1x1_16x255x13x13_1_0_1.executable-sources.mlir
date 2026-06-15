module @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_0_generic_16x169x1024_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_0_generic_16x169x1024_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x1024x169xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x169x1024xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 1024, 169], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x1024x169xf32>> -> tensor<16x1024x169xf32>
          %3 = tensor.empty() : tensor<16x169x1024xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<16x1024x169xf32>) outs(%3 : tensor<16x169x1024xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<16x169x1024xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [16, 169, 1024], strides = [1, 1, 1] : tensor<16x169x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x169x1024xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_1_generic_1024x255_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_1_generic_1024x255_f32() {
          %c0 = arith.constant 0 : index
          %c11075584 = arith.constant 11075584 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<255x1024xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c11075584) : !flow.dispatch.tensor<writeonly:tensor<1024x255xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<255x1024xf32>> -> tensor<255x1024xf32>
          %3 = tensor.empty() : tensor<1024x255xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<255x1024xf32>) outs(%3 : tensor<1024x255xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<1024x255xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [1024, 255], strides = [1, 1] : tensor<1024x255xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x255xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_2 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_16x13x13x255x1x1x1024_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_16x13x13x255x1x1x1024_f32() {
          %c0 = arith.constant 0 : index
          %c11075584 = arith.constant 11075584 : index
          %c12120064 = arith.constant 12120064 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x13x13x1024xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c11075584) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1x1024x255xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c12120064) : !flow.dispatch.tensor<writeonly:tensor<16x13x13x255xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [16, 13, 13, 1024], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x13x13x1024xf32>> -> tensor<16x13x13x1024xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [1, 1, 1024, 255], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x1024x255xf32>> -> tensor<1x1x1024x255xf32>
          %5 = tensor.empty() : tensor<16x13x13x255xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<16x13x13x255xf32>) -> tensor<16x13x13x255xf32>
          %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<16x13x13x1024xf32>, tensor<1x1x1024x255xf32>) outs(%6 : tensor<16x13x13x255xf32>) -> tensor<16x13x13x255xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [16, 13, 13, 255], strides = [1, 1, 1, 1] : tensor<16x13x13x255xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x13x13x255xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_3 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_3_generic_16x255x169_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_3_generic_16x255x169_f32() {
          %c12120064 = arith.constant 12120064 : index
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c12120064) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x169x255xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x255x169xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 169, 255], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x169x255xf32>> -> tensor<16x169x255xf32>
          %3 = tensor.empty() : tensor<16x255x169xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<16x169x255xf32>) outs(%3 : tensor<16x255x169xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<16x255x169xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [16, 255, 169], strides = [1, 1, 1] : tensor<16x255x169xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x255x169xf32>>
          return
        }
      }
    }
  }
  func.func @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c14878144 = arith.constant 14878144 : index
    %c11075584 = arith.constant 11075584 : index
    %c1044480 = arith.constant 1044480 : index
    %c2758080 = arith.constant 2758080 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c255 = arith.constant 255 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %c1024 = arith.constant 1024 : index
    %c13 = arith.constant 13 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c16, %c1024, %c13, %c13]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<16x1024x13x13xf32> in !stream.resource<external>{%c11075584}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c255, %c1024, %c1, %c1]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<255x1024x1x1xf32> in !stream.resource<external>{%c1044480}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c2758080}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c14878144} => !stream.timepoint
    %3 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c11075584}, %1 as %arg3: !stream.resource<external>{%c1044480}, %2 as %arg4: !stream.resource<external>{%c2758080}, %result as %arg5: !stream.resource<transient>{%c14878144}) {
      stream.cmd.concurrent {
        stream.cmd.dispatch @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_0::@cuda_nvptx_fb::@f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_0_generic_16x169x1024_f32 {
          ro %arg2[%c0 for %c11075584] : !stream.resource<external>{%c11075584},
          wo %arg5[%c0 for %c14878144] : !stream.resource<transient>{%c14878144}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.dispatch @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_1::@cuda_nvptx_fb::@f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_1_generic_1024x255_f32 {
          ro %arg3[%c0 for %c1044480] : !stream.resource<external>{%c1044480},
          wo %arg5[%c0 for %c14878144] : !stream.resource<transient>{%c14878144}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      }
      stream.cmd.dispatch @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_2::@cuda_nvptx_fb::@f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_16x13x13x255x1x1x1024_f32 {
        ro %arg5[%c0 for %c14878144] : !stream.resource<transient>{%c14878144},
        wo %arg5[%c0 for %c14878144] : !stream.resource<transient>{%c14878144}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_3::@cuda_nvptx_fb::@f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_3_generic_16x255x169_f32 {
        ro %arg5[%c0 for %c14878144] : !stream.resource<transient>{%c14878144},
        wo %arg4[%c0 for %c2758080] : !stream.resource<external>{%c2758080}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %4 = stream.resource.dealloca await(%3) => %result : !stream.resource<transient>{%c14878144} => !stream.timepoint
    %5 = stream.timepoint.await %4 => %2 : !stream.resource<external>{%c2758080}
    %6 = stream.tensor.export %5 : tensor<16x255x13x13xf32> in !stream.resource<external>{%c2758080} -> !hal.buffer_view
    return %6 : !hal.buffer_view
  }
}
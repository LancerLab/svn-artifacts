module @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_0_generic_64x50176x3_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_0_generic_64x50176x3_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x3x50176xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x50176x3xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 3, 50176], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x3x50176xf32>> -> tensor<64x3x50176xf32>
          %3 = tensor.empty() : tensor<64x50176x3xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<64x3x50176xf32>) outs(%3 : tensor<64x50176x3xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<64x50176x3xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [64, 50176, 3], strides = [1, 1, 1] : tensor<64x50176x3xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x50176x3xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_1_generic_49x3x64_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_1_generic_49x3x64_f32() {
          %c0 = arith.constant 0 : index
          %c38535168 = arith.constant 38535168 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x3x49xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c38535168) : !flow.dispatch.tensor<writeonly:tensor<49x3x64xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 3, 49], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x3x49xf32>> -> tensor<64x3x49xf32>
          %3 = tensor.empty() : tensor<49x3x64xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<64x3x49xf32>) outs(%3 : tensor<49x3x64xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<49x3x64xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [49, 3, 64], strides = [1, 1, 1] : tensor<49x3x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<49x3x64xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_2 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_2 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_2() {
          %c0 = arith.constant 0 : index
          %c38572800 = arith.constant 38572800 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x224x224x3xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c38572800) : !flow.dispatch.tensor<readwrite:tensor<64x230x230x3xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [64, 224, 224, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x224x224x3xf32>> -> tensor<64x224x224x3xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 3, 3, 0], sizes = [64, 224, 224, 3], strides = [1, 1, 1, 1] : tensor<64x224x224x3xf32> -> !flow.dispatch.tensor<readwrite:tensor<64x230x230x3xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_3 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_64x112x112x64x7x7x3_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_64x112x112x64x7x7x3_f32() {
          %c38572800 = arith.constant 38572800 : index
          %c38535168 = arith.constant 38535168 : index
          %c79200000 = arith.constant 79200000 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c38572800) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x230x230x3xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c38535168) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<7x7x3x64xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c79200000) : !flow.dispatch.tensor<writeonly:tensor<64x112x112x64xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [64, 230, 230, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x230x230x3xf32>> -> tensor<64x230x230x3xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [7, 7, 3, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<7x7x3x64xf32>> -> tensor<7x7x3x64xf32>
          %5 = tensor.empty() : tensor<64x112x112x64xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<64x112x112x64xf32>) -> tensor<64x112x112x64xf32>
          %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<64x230x230x3xf32>, tensor<7x7x3x64xf32>) outs(%6 : tensor<64x112x112x64xf32>) -> tensor<64x112x112x64xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [64, 112, 112, 64], strides = [1, 1, 1, 1] : tensor<64x112x112x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x112x112x64xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_4 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_4_generic_64x64x12544_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_4_generic_64x64x12544_f32() {
          %c79200000 = arith.constant 79200000 : index
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c79200000) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x12544x64xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x64x12544xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 12544, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x12544x64xf32>> -> tensor<64x12544x64xf32>
          %3 = tensor.empty() : tensor<64x64x12544xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<64x12544x64xf32>) outs(%3 : tensor<64x64x12544xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<64x64x12544xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [64, 64, 12544], strides = [1, 1, 1] : tensor<64x64x12544xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x64x12544xf32>>
          return
        }
      }
    }
  }
  func.func @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c284720896 = arith.constant 284720896 : index
    %c38572800 = arith.constant 38572800 : index
    %c38535168 = arith.constant 38535168 : index
    %c37632 = arith.constant 37632 : index
    %c40627200 = arith.constant 40627200 : index
    %c205520896 = arith.constant 205520896 : index
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c3 = arith.constant 3 : index
    %c224 = arith.constant 224 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c64, %c3, %c224, %c224]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<64x3x224x224xf32> in !stream.resource<external>{%c38535168}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c64, %c3, %c7, %c7]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<64x3x7x7xf32> in !stream.resource<external>{%c37632}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c205520896}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c284720896} => !stream.timepoint
    %3 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c38535168}, %1 as %arg3: !stream.resource<external>{%c37632}, %2 as %arg4: !stream.resource<external>{%c205520896}, %result as %arg5: !stream.resource<transient>{%c284720896}) {
      stream.cmd.concurrent {
        stream.cmd.dispatch @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_0::@cuda_nvptx_fb::@f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_0_generic_64x50176x3_f32 {
          ro %arg2[%c0 for %c38535168] : !stream.resource<external>{%c38535168},
          wo %arg5[%c0 for %c284720896] : !stream.resource<transient>{%c284720896}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.dispatch @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_1::@cuda_nvptx_fb::@f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_1_generic_49x3x64_f32 {
          ro %arg3[%c0 for %c37632] : !stream.resource<external>{%c37632},
          wo %arg5[%c0 for %c284720896] : !stream.resource<transient>{%c284720896}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.fill %c0_i8, %arg5[%c38572800 for %c40627200] : i8 -> !stream.resource<transient>{%c284720896}
      }
      stream.cmd.dispatch @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_2::@cuda_nvptx_fb::@f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_2 {
        ro %arg5[%c0 for %c284720896] : !stream.resource<transient>{%c284720896},
        rw %arg5[%c0 for %c284720896] : !stream.resource<transient>{%c284720896}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_3::@cuda_nvptx_fb::@f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_64x112x112x64x7x7x3_f32 {
        ro %arg5[%c0 for %c284720896] : !stream.resource<transient>{%c284720896},
        wo %arg5[%c0 for %c284720896] : !stream.resource<transient>{%c284720896}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_4::@cuda_nvptx_fb::@f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_dispatch_4_generic_64x64x12544_f32 {
        ro %arg5[%c0 for %c284720896] : !stream.resource<transient>{%c284720896},
        wo %arg4[%c0 for %c205520896] : !stream.resource<external>{%c205520896}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %4 = stream.resource.dealloca await(%3) => %result : !stream.resource<transient>{%c284720896} => !stream.timepoint
    %5 = stream.timepoint.await %4 => %2 : !stream.resource<external>{%c205520896}
    %6 = stream.tensor.export %5 : tensor<64x64x112x112xf32> in !stream.resource<external>{%c205520896} -> !hal.buffer_view
    return %6 : !hal.buffer_view
  }
}
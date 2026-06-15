module @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_0_generic_16x50176x64_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_0_generic_16x50176x64_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x64x50176xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x50176x64xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 64, 50176], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x64x50176xf32>> -> tensor<16x64x50176xf32>
          %3 = tensor.empty() : tensor<16x50176x64xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<16x64x50176xf32>) outs(%3 : tensor<16x50176x64xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<16x50176x64xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [16, 50176, 64], strides = [1, 1, 1] : tensor<16x50176x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x50176x64xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_1_generic_49x64x128_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_1_generic_49x64x128_f32() {
          %c0 = arith.constant 0 : index
          %c205520896 = arith.constant 205520896 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x64x49xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c205520896) : !flow.dispatch.tensor<writeonly:tensor<49x64x128xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [128, 64, 49], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x64x49xf32>> -> tensor<128x64x49xf32>
          %3 = tensor.empty() : tensor<49x64x128xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<128x64x49xf32>) outs(%3 : tensor<49x64x128xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<49x64x128xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [49, 64, 128], strides = [1, 1, 1] : tensor<49x64x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<49x64x128xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_2 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_2 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_2() {
          %c0 = arith.constant 0 : index
          %c207126528 = arith.constant 207126528 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x224x224x64xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c207126528) : !flow.dispatch.tensor<readwrite:tensor<16x230x230x64xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [16, 224, 224, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x224x224x64xf32>> -> tensor<16x224x224x64xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 3, 3, 0], sizes = [16, 224, 224, 64], strides = [1, 1, 1, 1] : tensor<16x224x224x64xf32> -> !flow.dispatch.tensor<readwrite:tensor<16x230x230x64xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_3 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_16x112x112x128x7x7x64_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_16x112x112x128x7x7x64_f32() {
          %c207126528 = arith.constant 207126528 : index
          %c205520896 = arith.constant 205520896 : index
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c207126528) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x230x230x64xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c205520896) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<7x7x64x128xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x112x112x128xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [16, 230, 230, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x230x230x64xf32>> -> tensor<16x230x230x64xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [7, 7, 64, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<7x7x64x128xf32>> -> tensor<7x7x64x128xf32>
          %5 = tensor.empty() : tensor<16x112x112x128xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<16x112x112x128xf32>) -> tensor<16x112x112x128xf32>
          %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<16x230x230x64xf32>, tensor<7x7x64x128xf32>) outs(%6 : tensor<16x112x112x128xf32>) -> tensor<16x112x112x128xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [16, 112, 112, 128], strides = [1, 1, 1, 1] : tensor<16x112x112x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x112x112x128xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_4 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_4_generic_16x128x12544_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_4_generic_16x128x12544_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x12544x128xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x128x12544xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 12544, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x12544x128xf32>> -> tensor<16x12544x128xf32>
          %3 = tensor.empty() : tensor<16x128x12544xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<16x12544x128xf32>) outs(%3 : tensor<16x128x12544xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<16x128x12544xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [16, 128, 12544], strides = [1, 1, 1] : tensor<16x128x12544xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x128x12544xf32>>
          return
        }
      }
    }
  }
  func.func @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c423804928 = arith.constant 423804928 : index
    %c207126528 = arith.constant 207126528 : index
    %c205520896 = arith.constant 205520896 : index
    %c1605632 = arith.constant 1605632 : index
    %c216678400 = arith.constant 216678400 : index
    %c102760448 = arith.constant 102760448 : index
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c128 = arith.constant 128 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c224 = arith.constant 224 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c16, %c64, %c224, %c224]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<16x64x224x224xf32> in !stream.resource<external>{%c205520896}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c128, %c64, %c7, %c7]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<128x64x7x7xf32> in !stream.resource<external>{%c1605632}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c102760448}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c423804928} => !stream.timepoint
    %3 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c205520896}, %1 as %arg3: !stream.resource<external>{%c1605632}, %2 as %arg4: !stream.resource<external>{%c102760448}, %result as %arg5: !stream.resource<transient>{%c423804928}) {
      stream.cmd.concurrent {
        stream.cmd.dispatch @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_0::@cuda_nvptx_fb::@f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_0_generic_16x50176x64_f32 {
          ro %arg2[%c0 for %c205520896] : !stream.resource<external>{%c205520896},
          wo %arg5[%c0 for %c423804928] : !stream.resource<transient>{%c423804928}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.dispatch @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_1::@cuda_nvptx_fb::@f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_1_generic_49x64x128_f32 {
          ro %arg3[%c0 for %c1605632] : !stream.resource<external>{%c1605632},
          wo %arg5[%c0 for %c423804928] : !stream.resource<transient>{%c423804928}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.fill %c0_i8, %arg5[%c207126528 for %c216678400] : i8 -> !stream.resource<transient>{%c423804928}
      }
      stream.cmd.dispatch @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_2::@cuda_nvptx_fb::@f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_2 {
        ro %arg5[%c0 for %c423804928] : !stream.resource<transient>{%c423804928},
        rw %arg5[%c0 for %c423804928] : !stream.resource<transient>{%c423804928}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_3::@cuda_nvptx_fb::@f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_16x112x112x128x7x7x64_f32 {
        ro %arg5[%c0 for %c423804928] : !stream.resource<transient>{%c423804928},
        wo %arg5[%c0 for %c423804928] : !stream.resource<transient>{%c423804928}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_4::@cuda_nvptx_fb::@f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_4_generic_16x128x12544_f32 {
        ro %arg5[%c0 for %c423804928] : !stream.resource<transient>{%c423804928},
        wo %arg4[%c0 for %c102760448] : !stream.resource<external>{%c102760448}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %4 = stream.resource.dealloca await(%3) => %result : !stream.resource<transient>{%c423804928} => !stream.timepoint
    %5 = stream.timepoint.await %4 => %2 : !stream.resource<external>{%c102760448}
    %6 = stream.tensor.export %5 : tensor<16x128x112x112xf32> in !stream.resource<external>{%c102760448} -> !hal.buffer_view
    return %6 : !hal.buffer_view
  }
}
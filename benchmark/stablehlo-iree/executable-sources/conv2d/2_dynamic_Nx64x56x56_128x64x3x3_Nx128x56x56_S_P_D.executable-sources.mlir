module @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_0_generic_Dx56x56x64_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 1, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_0_generic_Dx56x56x64_f32() {
          %c294912 = arith.constant 294912 : index
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = arith.index_castui %0 : i32 to index
          %2 = flow.dispatch.workload.ordinal %1, 0 : index
          %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x64x56x56xf32>>{%2}
          %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c294912) : !flow.dispatch.tensor<writeonly:tensor<?x56x56x64xf32>>{%2}
          %5 = flow.dispatch.tensor.load %3, offsets = [0, 0, 0, 0], sizes = [%2, 64, 56, 56], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x64x56x56xf32>>{%2} -> tensor<?x64x56x56xf32>
          %6 = tensor.empty(%2) : tensor<?x56x56x64xf32>
          %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5 : tensor<?x64x56x56xf32>) outs(%6 : tensor<?x56x56x64xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<?x56x56x64xf32>
          flow.dispatch.tensor.store %7, %4, offsets = [0, 0, 0, 0], sizes = [%2, 56, 56, 64], strides = [1, 1, 1, 1] : tensor<?x56x56x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x56x56x64xf32>>{%2}
          return
        }
      }
    }
  }
  hal.executable private @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_1_generic_9x64x128_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_1_generic_9x64x128_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x64x9xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<9x64x128xf32>>
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
  hal.executable private @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_2 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_2 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_2() {
          %c294912 = arith.constant 294912 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = arith.index_castui %0 : i32 to index
          %3 = arith.index_castui %1 : i32 to index
          %4 = flow.dispatch.workload.ordinal %3, 0 : index
          %5 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c294912) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x56x56x64xf32>>{%4}
          %6 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%2) : !flow.dispatch.tensor<readwrite:tensor<?x58x58x64xf32>>{%4}
          %7 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0, 0], sizes = [%4, 56, 56, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x56x56x64xf32>>{%4} -> tensor<?x56x56x64xf32>
          flow.dispatch.tensor.store %7, %6, offsets = [0, 1, 1, 0], sizes = [%4, 56, 56, 64], strides = [1, 1, 1, 1] : tensor<?x56x56x64xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x58x58x64xf32>>{%4}
          return
        }
      }
    }
  }
  hal.executable private @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_3 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_Dx56x56x128x3x3x64_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 3, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_Dx56x56x128x3x3x64_f32() {
          %cst = arith.constant 0.000000e+00 : f32
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = hal.interface.constant.load[2] : i32
          %3 = arith.index_castui %0 : i32 to index
          %4 = arith.index_castui %1 : i32 to index
          %5 = arith.index_castui %2 : i32 to index
          %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x64x128xf32>>
          %7 = flow.dispatch.workload.ordinal %5, 0 : index
          %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%3) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x58x58x64xf32>>{%7}
          %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%4) : !flow.dispatch.tensor<writeonly:tensor<?x56x56x128xf32>>{%7}
          %10 = flow.dispatch.tensor.load %8, offsets = [0, 0, 0, 0], sizes = [%7, 58, 58, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x58x58x64xf32>>{%7} -> tensor<?x58x58x64xf32>
          %11 = flow.dispatch.tensor.load %6, offsets = [0, 0, 0, 0], sizes = [3, 3, 64, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x64x128xf32>> -> tensor<3x3x64x128xf32>
          %12 = tensor.empty(%7) : tensor<?x56x56x128xf32>
          %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<?x56x56x128xf32>) -> tensor<?x56x56x128xf32>
          %14 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%10, %11 : tensor<?x58x58x64xf32>, tensor<3x3x64x128xf32>) outs(%13 : tensor<?x56x56x128xf32>) -> tensor<?x56x56x128xf32>
          flow.dispatch.tensor.store %14, %9, offsets = [0, 0, 0, 0], sizes = [%7, 56, 56, 128], strides = [1, 1, 1, 1] : tensor<?x56x56x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x56x56x128xf32>>{%7}
          return
        }
      }
    }
  }
  hal.executable private @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_4 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_4_generic_Dx128x56x56_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_4_generic_Dx128x56x56_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = arith.index_castui %0 : i32 to index
          %3 = arith.index_castui %1 : i32 to index
          %4 = flow.dispatch.workload.ordinal %3, 0 : index
          %5 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%2) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x56x56x128xf32>>{%4}
          %6 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x128x56x56xf32>>{%4}
          %7 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0, 0], sizes = [%4, 56, 56, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x56x56x128xf32>>{%4} -> tensor<?x56x56x128xf32>
          %8 = tensor.empty(%4) : tensor<?x128x56x56xf32>
          %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7 : tensor<?x56x56x128xf32>) outs(%8 : tensor<?x128x56x56xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<?x128x56x56xf32>
          flow.dispatch.tensor.store %9, %6, offsets = [0, 0, 0, 0], sizes = [%4, 128, 56, 56], strides = [1, 1, 1, 1] : tensor<?x128x56x56xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x128x56x56xf32>>{%4}
          return
        }
      }
    }
  }
  func.func @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c802816 = arith.constant 802816 : index
    %c294912 = arith.constant 294912 : index
    %c861184 = arith.constant 861184 : index
    %c1605632 = arith.constant 1605632 : index
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c128 = arith.constant 128 : index
    %c56 = arith.constant 56 : index
    %c64 = arith.constant 64 : index
    %c1_i32 = arith.constant 1 : i32
    %c553648160_i32 = arith.constant 553648160 : i32
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%0, %c64, %c56, %c56]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = arith.muli %0, %c802816 : index
    %2 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<?x64x56x56xf32>{%0} in !stream.resource<external>{%1}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c128, %c64, %c3, %c3]) type(%c553648160_i32) encoding(%c1_i32)
    %3 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<128x64x3x3xf32> in !stream.resource<external>{%c294912}
    %4 = arith.muli %0, %c861184 : index
    %5 = arith.muli %0, %c1605632 : index
    %6 = stream.resource.alloc uninitialized : !stream.resource<external>{%5}
    %7 = util.align %1, %c64 : index
    %8 = arith.addi %7, %c294912 : index
    %9 = util.align %4, %c64 : index
    %10 = arith.addi %8, %9 : index
    %11 = util.align %5, %c64 : index
    %12 = arith.addi %10, %11 : index
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%12} => !stream.timepoint
    %13 = arith.index_castui %0 : index to i32
    %14 = arith.index_castui %8 : index to i32
    %15 = arith.index_castui %10 : index to i32
    %16 = stream.cmd.execute await(%result_timepoint) => with(%2 as %arg2: !stream.resource<external>{%1}, %3 as %arg3: !stream.resource<external>{%c294912}, %6 as %arg4: !stream.resource<external>{%5}, %result as %arg5: !stream.resource<transient>{%12}) {
      stream.cmd.concurrent {
        stream.cmd.dispatch @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_0::@cuda_nvptx_fb::@f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_0_generic_Dx56x56x64_f32[%0](%13 : i32) {
          ro %arg2[%c0 for %1] : !stream.resource<external>{%1},
          wo %arg5[%c0 for %12] : !stream.resource<transient>{%12}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.dispatch @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_1::@cuda_nvptx_fb::@f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_1_generic_9x64x128_f32 {
          ro %arg3[%c0 for %c294912] : !stream.resource<external>{%c294912},
          wo %arg5[%c0 for %12] : !stream.resource<transient>{%12}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.fill %c0_i8, %arg5[%8 for %4] : i8 -> !stream.resource<transient>{%12}
      }
      stream.cmd.dispatch @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_2::@cuda_nvptx_fb::@f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_2[%0](%14, %13 : i32, i32) {
        ro %arg5[%c0 for %12] : !stream.resource<transient>{%12},
        rw %arg5[%c0 for %12] : !stream.resource<transient>{%12}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_3::@cuda_nvptx_fb::@f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_Dx56x56x128x3x3x64_f32[%0](%14, %15, %13 : i32, i32, i32) {
        ro %arg5[%c0 for %12] : !stream.resource<transient>{%12},
        wo %arg5[%c0 for %12] : !stream.resource<transient>{%12}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_4::@cuda_nvptx_fb::@f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_4_generic_Dx128x56x56_f32[%0](%15, %13 : i32, i32) {
        ro %arg5[%c0 for %12] : !stream.resource<transient>{%12},
        wo %arg4[%c0 for %5] : !stream.resource<external>{%5}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %17 = stream.resource.dealloca await(%16) => %result : !stream.resource<transient>{%12} => !stream.timepoint
    %18 = stream.timepoint.await %17 => %6 : !stream.resource<external>{%5}
    %19 = stream.tensor.export %18 : tensor<?x128x56x56xf32>{%0} in !stream.resource<external>{%5} -> !hal.buffer_view
    return %19 : !hal.buffer_view
  }
}
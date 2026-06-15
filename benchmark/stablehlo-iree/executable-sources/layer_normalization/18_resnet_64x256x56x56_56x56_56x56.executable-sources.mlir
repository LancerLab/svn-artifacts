module @f_18_resnet_64x256x56x56_56x56_56x56 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_18_resnet_64x256x56x56_56x56_56x56_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_18_resnet_64x256x56x56_56x56_56x56_dispatch_0_generic_16384x3136_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_18_resnet_64x256x56x56_56x56_56x56_dispatch_0_generic_16384x3136_f32() {
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = arith.index_castui %0 {stream.alignment = 65536 : index, stream.values = [0 : index, 65536 : index]} : i32 to index
          %3 = arith.index_castui %1 {stream.alignment = 65536 : index, stream.values = [0 : index, 205586432 : index]} : i32 to index
          %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%2) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16384x3136xf32>>
          %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%3) : !flow.dispatch.tensor<readwrite:tensor<16384xf32>>
          %6 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [16384, 3136], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16384x3136xf32>> -> tensor<16384x3136xf32>
          %7 = flow.dispatch.tensor.load %5, offsets = [0], sizes = [16384], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<16384xf32>> -> tensor<16384xf32>
          %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%6 : tensor<16384x3136xf32>) outs(%7 : tensor<16384xf32>) {
          ^bb0(%in: f32, %out: f32):
            %9 = arith.addf %out, %in : f32
            linalg.yield %9 : f32
          } -> tensor<16384xf32>
          flow.dispatch.tensor.store %8, %5, offsets = [0], sizes = [16384], strides = [1] : tensor<16384xf32> -> !flow.dispatch.tensor<readwrite:tensor<16384xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_18_resnet_64x256x56x56_56x56_56x56_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_18_resnet_64x256x56x56_56x56_56x56_dispatch_1_generic_51380224_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_18_resnet_64x256x56x56_56x56_56x56_dispatch_1_generic_51380224_f32() {
          %c0 = arith.constant 0 : index
          %c65536 = arith.constant 65536 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<51380224xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c65536) : !flow.dispatch.tensor<writeonly:tensor<51380224xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [51380224], strides = [1] : !flow.dispatch.tensor<readonly:tensor<51380224xf32>> -> tensor<51380224xf32>
          %3 = tensor.empty() : tensor<51380224xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2 : tensor<51380224xf32>) outs(%3 : tensor<51380224xf32>) {
          ^bb0(%in: f32, %out: f32):
            %5 = arith.mulf %in, %in : f32
            linalg.yield %5 : f32
          } -> tensor<51380224xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0], sizes = [51380224], strides = [1] : tensor<51380224xf32> -> !flow.dispatch.tensor<writeonly:tensor<51380224xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_18_resnet_64x256x56x56_56x56_56x56_dispatch_3 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_18_resnet_64x256x56x56_56x56_56x56_dispatch_3_generic_64x256x56x56_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer, ReadOnly>, <4, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_18_resnet_64x256x56x56_56x56_56x56_dispatch_3_generic_64x256x56x56_f32() {
          %c0 = arith.constant 0 : index
          %c205586432 = arith.constant 205586432 : index
          %cst = arith.constant 3.136000e+03 : f32
          %cst_0 = arith.constant 9.99999974E-6 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x256x56x56xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x256xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c205586432) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x256xf32>>
          %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<56x56xf32>>
          %4 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<56x56xf32>>
          %5 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x256x56x56xf32>>
          %6 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [64, 256, 56, 56], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x256x56x56xf32>> -> tensor<64x256x56x56xf32>
          %7 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [64, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x256xf32>> -> tensor<64x256xf32>
          %8 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [64, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x256xf32>> -> tensor<64x256xf32>
          %9 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [56, 56], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<56x56xf32>> -> tensor<56x56xf32>
          %10 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [56, 56], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<56x56xf32>> -> tensor<56x56xf32>
          %11 = tensor.empty() : tensor<64x256x56x56xf32>
          %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6, %7, %8, %9, %10 : tensor<64x256x56x56xf32>, tensor<64x256xf32>, tensor<64x256xf32>, tensor<56x56xf32>, tensor<56x56xf32>) outs(%11 : tensor<64x256x56x56xf32>) {
          ^bb0(%in: f32, %in_1: f32, %in_2: f32, %in_3: f32, %in_4: f32, %out: f32):
            %13 = arith.divf %in_1, %cst : f32
            %14 = arith.mulf %13, %13 : f32
            %15 = arith.divf %in_2, %cst : f32
            %16 = arith.subf %15, %14 : f32
            %17 = arith.addf %16, %cst_0 : f32
            %18 = math.rsqrt %17 : f32
            %19 = arith.subf %in, %13 : f32
            %20 = arith.mulf %19, %18 : f32
            %21 = arith.mulf %20, %in_3 : f32
            %22 = arith.addf %21, %in_4 : f32
            linalg.yield %22 : f32
          } -> tensor<64x256x56x56xf32>
          flow.dispatch.tensor.store %12, %5, offsets = [0, 0, 0, 0], sizes = [64, 256, 56, 56], strides = [1, 1, 1, 1] : tensor<64x256x56x56xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x256x56x56xf32>>
          return
        }
      }
    }
  }
  func.func @f_18_resnet_64x256x56x56_56x56_56x56(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c205586432_i32 = arith.constant 205586432 : i32
    %c65536_i32 = arith.constant 65536 : i32
    %c0_i32 = arith.constant 0 : i32
    %c205651968 = arith.constant 205651968 : index
    %c205586432 = arith.constant 205586432 : index
    %c205520896 = arith.constant 205520896 : index
    %c12544 = arith.constant 12544 : index
    %c65536 = arith.constant 65536 : index
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c56 = arith.constant 56 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c64, %c256, %c56, %c56]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<64x256x56x56xf32> in !stream.resource<external>{%c205520896}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c56, %c56]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<56x56xf32> in !stream.resource<external>{%c12544}
    hal.buffer_view.assert<%arg2 : !hal.buffer_view> message("input 2") shape([%c56, %c56]) type(%c553648160_i32) encoding(%c1_i32)
    %2 = stream.tensor.import %arg2 : !hal.buffer_view -> tensor<56x56xf32> in !stream.resource<external>{%c12544}
    %3 = stream.resource.alloc uninitialized : !stream.resource<external>{%c205520896}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c205651968} => !stream.timepoint
    %4 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg3: !stream.resource<external>{%c205520896}, %1 as %arg4: !stream.resource<external>{%c12544}, %2 as %arg5: !stream.resource<external>{%c12544}, %3 as %arg6: !stream.resource<external>{%c205520896}, %result as %arg7: !stream.resource<transient>{%c205651968}) {
      stream.cmd.fill %c0_i8, %arg7[%c0 for %c65536] : i8 -> !stream.resource<transient>{%c205651968}
      stream.cmd.concurrent {
        stream.cmd.dispatch @f_18_resnet_64x256x56x56_56x56_56x56_dispatch_0::@cuda_nvptx_fb::@f_18_resnet_64x256x56x56_56x56_56x56_dispatch_0_generic_16384x3136_f32(%c0_i32, %c0_i32 : i32, i32) {
          ro %arg3[%c0 for %c205520896] : !stream.resource<external>{%c205520896},
          rw %arg7[%c0 for %c205651968] : !stream.resource<transient>{%c205651968}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.dispatch @f_18_resnet_64x256x56x56_56x56_56x56_dispatch_1::@cuda_nvptx_fb::@f_18_resnet_64x256x56x56_56x56_56x56_dispatch_1_generic_51380224_f32 {
          ro %arg3[%c0 for %c205520896] : !stream.resource<external>{%c205520896},
          wo %arg7[%c0 for %c205651968] : !stream.resource<transient>{%c205651968}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.fill %c0_i8, %arg7[%c205586432 for %c65536] : i8 -> !stream.resource<transient>{%c205651968}
      }
      stream.cmd.dispatch @f_18_resnet_64x256x56x56_56x56_56x56_dispatch_0::@cuda_nvptx_fb::@f_18_resnet_64x256x56x56_56x56_56x56_dispatch_0_generic_16384x3136_f32(%c65536_i32, %c205586432_i32 : i32, i32) {
        ro %arg7[%c0 for %c205651968] : !stream.resource<transient>{%c205651968},
        rw %arg7[%c0 for %c205651968] : !stream.resource<transient>{%c205651968}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_18_resnet_64x256x56x56_56x56_56x56_dispatch_3::@cuda_nvptx_fb::@f_18_resnet_64x256x56x56_56x56_56x56_dispatch_3_generic_64x256x56x56_f32 {
        ro %arg3[%c0 for %c205520896] : !stream.resource<external>{%c205520896},
        ro %arg7[%c0 for %c205651968] : !stream.resource<transient>{%c205651968},
        ro %arg4[%c0 for %c12544] : !stream.resource<external>{%c12544},
        ro %arg5[%c0 for %c12544] : !stream.resource<external>{%c12544},
        wo %arg6[%c0 for %c205520896] : !stream.resource<external>{%c205520896}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>, #hal.interface.binding<0, 4>]}
    } => !stream.timepoint
    %5 = stream.resource.dealloca await(%4) => %result : !stream.resource<transient>{%c205651968} => !stream.timepoint
    %6 = stream.timepoint.await %5 => %3 : !stream.resource<external>{%c205520896}
    %7 = stream.tensor.export %6 : tensor<64x256x56x56xf32> in !stream.resource<external>{%c205520896} -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
}
module @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_0_generic_32x50176x3_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_0_generic_32x50176x3_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x3x50176xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x50176x3xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [32, 3, 50176], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x3x50176xf32>> -> tensor<32x3x50176xf32>
          %3 = tensor.empty() : tensor<32x50176x3xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<32x3x50176xf32>) outs(%3 : tensor<32x50176x3xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<32x50176x3xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [32, 50176, 3], strides = [1, 1, 1] : tensor<32x50176x3xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x50176x3xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_1_generic_256x3x768_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_1_generic_256x3x768_f32() {
          %c0 = arith.constant 0 : index
          %c19267584 = arith.constant 19267584 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<768x3x256xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c19267584) : !flow.dispatch.tensor<writeonly:tensor<256x3x768xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [768, 3, 256], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<768x3x256xf32>> -> tensor<768x3x256xf32>
          %3 = tensor.empty() : tensor<256x3x768xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<768x3x256xf32>) outs(%3 : tensor<256x3x768xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<256x3x768xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [256, 3, 768], strides = [1, 1, 1] : tensor<256x3x768xf32> -> !flow.dispatch.tensor<writeonly:tensor<256x3x768xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_2 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_2_conv_2d_nhwc_hwcf_32x14x14x768x16x16x3_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_2_conv_2d_nhwc_hwcf_32x14x14x768x16x16x3_f32() {
          %c0 = arith.constant 0 : index
          %c19267584 = arith.constant 19267584 : index
          %c21626880 = arith.constant 21626880 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x224x224x3xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c19267584) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x16x3x768xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c21626880) : !flow.dispatch.tensor<writeonly:tensor<32x14x14x768xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [32, 224, 224, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x224x224x3xf32>> -> tensor<32x224x224x3xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [16, 16, 3, 768], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x16x3x768xf32>> -> tensor<16x16x3x768xf32>
          %5 = tensor.empty() : tensor<32x14x14x768xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<32x14x14x768xf32>) -> tensor<32x14x14x768xf32>
          %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<16> : tensor<2xi64>} ins(%3, %4 : tensor<32x224x224x3xf32>, tensor<16x16x3x768xf32>) outs(%6 : tensor<32x14x14x768xf32>) -> tensor<32x14x14x768xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [32, 14, 14, 768], strides = [1, 1, 1, 1] : tensor<32x14x14x768xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x14x14x768xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_3 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_3_generic_32x768x196_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_3_generic_32x768x196_f32() {
          %c21626880 = arith.constant 21626880 : index
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c21626880) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x196x768xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x768x196xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [32, 196, 768], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x196x768xf32>> -> tensor<32x196x768xf32>
          %3 = tensor.empty() : tensor<32x768x196xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<32x196x768xf32>) outs(%3 : tensor<32x768x196xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<32x768x196xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [32, 768, 196], strides = [1, 1, 1] : tensor<32x768x196xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x768x196xf32>>
          return
        }
      }
    }
  }
  func.func @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c40894464 = arith.constant 40894464 : index
    %c19267584 = arith.constant 19267584 : index
    %c2359296 = arith.constant 2359296 : index
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c768 = arith.constant 768 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32 = arith.constant 32 : index
    %c3 = arith.constant 3 : index
    %c224 = arith.constant 224 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c32, %c3, %c224, %c224]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<32x3x224x224xf32> in !stream.resource<external>{%c19267584}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c768, %c3, %c16, %c16]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<768x3x16x16xf32> in !stream.resource<external>{%c2359296}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c19267584}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c40894464} => !stream.timepoint
    %3 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c19267584}, %1 as %arg3: !stream.resource<external>{%c2359296}, %2 as %arg4: !stream.resource<external>{%c19267584}, %result as %arg5: !stream.resource<transient>{%c40894464}) {
      stream.cmd.concurrent {
        stream.cmd.dispatch @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_0::@cuda_nvptx_fb::@f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_0_generic_32x50176x3_f32 {
          ro %arg2[%c0 for %c19267584] : !stream.resource<external>{%c19267584},
          wo %arg5[%c0 for %c40894464] : !stream.resource<transient>{%c40894464}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.dispatch @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_1::@cuda_nvptx_fb::@f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_1_generic_256x3x768_f32 {
          ro %arg3[%c0 for %c2359296] : !stream.resource<external>{%c2359296},
          wo %arg5[%c0 for %c40894464] : !stream.resource<transient>{%c40894464}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      }
      stream.cmd.dispatch @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_2::@cuda_nvptx_fb::@f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_2_conv_2d_nhwc_hwcf_32x14x14x768x16x16x3_f32 {
        ro %arg5[%c0 for %c40894464] : !stream.resource<transient>{%c40894464},
        wo %arg5[%c0 for %c40894464] : !stream.resource<transient>{%c40894464}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_3::@cuda_nvptx_fb::@f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_3_generic_32x768x196_f32 {
        ro %arg5[%c0 for %c40894464] : !stream.resource<transient>{%c40894464},
        wo %arg4[%c0 for %c19267584] : !stream.resource<external>{%c19267584}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %4 = stream.resource.dealloca await(%3) => %result : !stream.resource<transient>{%c40894464} => !stream.timepoint
    %5 = stream.timepoint.await %4 => %2 : !stream.resource<external>{%c19267584}
    %6 = stream.tensor.export %5 : tensor<32x768x14x14xf32> in !stream.resource<external>{%c19267584} -> !hal.buffer_view
    return %6 : !hal.buffer_view
  }
}
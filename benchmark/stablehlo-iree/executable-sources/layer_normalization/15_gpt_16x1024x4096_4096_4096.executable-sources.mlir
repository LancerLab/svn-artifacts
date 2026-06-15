module @f_15_gpt_16x1024x4096_4096_4096 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_15_gpt_16x1024x4096_4096_4096_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_15_gpt_16x1024x4096_4096_4096_dispatch_0_generic_16384x4096_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_15_gpt_16x1024x4096_4096_4096_dispatch_0_generic_16384x4096_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16384x4096xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<16384xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16384, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16384x4096xf32>> -> tensor<16384x4096xf32>
          %3 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [16384], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<16384xf32>> -> tensor<16384xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<16384x4096xf32>) outs(%3 : tensor<16384xf32>) {
          ^bb0(%in: f32, %out: f32):
            %5 = arith.mulf %in, %in : f32
            %6 = arith.addf %out, %5 : f32
            linalg.yield %6 : f32
          } -> tensor<16384xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0], sizes = [16384], strides = [1] : tensor<16384xf32> -> !flow.dispatch.tensor<readwrite:tensor<16384xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_15_gpt_16x1024x4096_4096_4096_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_15_gpt_16x1024x4096_4096_4096_dispatch_1_generic_16x1024x4096_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer, ReadOnly>, <4, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_15_gpt_16x1024x4096_4096_4096_dispatch_1_generic_16x1024x4096_f32() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %cst_0 = arith.constant 4.096000e+03 : f32
          %cst_1 = arith.constant 9.99999974E-6 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x1024x4096xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x1024xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096xf32>>
          %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096xf32>>
          %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x1024x4096xf32>>
          %5 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 1024, 4096], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x1024x4096xf32>> -> tensor<16x1024x4096xf32>
          %6 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x1024xf32>> -> tensor<16x1024xf32>
          %7 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [4096], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4096xf32>> -> tensor<4096xf32>
          %8 = flow.dispatch.tensor.load %3, offsets = [0], sizes = [4096], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4096xf32>> -> tensor<4096xf32>
          %9 = tensor.empty() : tensor<16x1024x4096xf32>
          %10 = tensor.empty() : tensor<16x1024xf32>
          %11 = linalg.fill ins(%cst : f32) outs(%10 : tensor<16x1024xf32>) -> tensor<16x1024xf32>
          %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%5 : tensor<16x1024x4096xf32>) outs(%11 : tensor<16x1024xf32>) {
          ^bb0(%in: f32, %out: f32):
            %14 = arith.addf %out, %in : f32
            linalg.yield %14 : f32
          } -> tensor<16x1024xf32>
          %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %12, %6, %7, %8 : tensor<16x1024x4096xf32>, tensor<16x1024xf32>, tensor<16x1024xf32>, tensor<4096xf32>, tensor<4096xf32>) outs(%9 : tensor<16x1024x4096xf32>) {
          ^bb0(%in: f32, %in_2: f32, %in_3: f32, %in_4: f32, %in_5: f32, %out: f32):
            %14 = arith.divf %in_2, %cst_0 : f32
            %15 = arith.mulf %14, %14 : f32
            %16 = arith.divf %in_3, %cst_0 : f32
            %17 = arith.subf %16, %15 : f32
            %18 = arith.addf %17, %cst_1 : f32
            %19 = math.rsqrt %18 : f32
            %20 = arith.subf %in, %14 : f32
            %21 = arith.mulf %20, %19 : f32
            %22 = arith.mulf %21, %in_4 : f32
            %23 = arith.addf %22, %in_5 : f32
            linalg.yield %23 : f32
          } -> tensor<16x1024x4096xf32>
          flow.dispatch.tensor.store %13, %4, offsets = [0, 0, 0], sizes = [16, 1024, 4096], strides = [1, 1, 1] : tensor<16x1024x4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x1024x4096xf32>>
          return
        }
      }
    }
  }
  func.func @f_15_gpt_16x1024x4096_4096_4096(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c268435456 = arith.constant 268435456 : index
    %c16384 = arith.constant 16384 : index
    %c65536 = arith.constant 65536 : index
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %c1024 = arith.constant 1024 : index
    %c4096 = arith.constant 4096 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c16, %c1024, %c4096]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<16x1024x4096xf32> in !stream.resource<external>{%c268435456}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c4096]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<4096xf32> in !stream.resource<external>{%c16384}
    hal.buffer_view.assert<%arg2 : !hal.buffer_view> message("input 2") shape([%c4096]) type(%c553648160_i32) encoding(%c1_i32)
    %2 = stream.tensor.import %arg2 : !hal.buffer_view -> tensor<4096xf32> in !stream.resource<external>{%c16384}
    %3 = stream.resource.alloc uninitialized : !stream.resource<external>{%c268435456}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c65536} => !stream.timepoint
    %4 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg3: !stream.resource<external>{%c268435456}, %1 as %arg4: !stream.resource<external>{%c16384}, %2 as %arg5: !stream.resource<external>{%c16384}, %3 as %arg6: !stream.resource<external>{%c268435456}, %result as %arg7: !stream.resource<transient>{%c65536}) {
      stream.cmd.fill %c0_i8, %arg7[%c0 for %c65536] : i8 -> !stream.resource<transient>{%c65536}
      stream.cmd.dispatch @f_15_gpt_16x1024x4096_4096_4096_dispatch_0::@cuda_nvptx_fb::@f_15_gpt_16x1024x4096_4096_4096_dispatch_0_generic_16384x4096_f32 {
        ro %arg3[%c0 for %c268435456] : !stream.resource<external>{%c268435456},
        rw %arg7[%c0 for %c65536] : !stream.resource<transient>{%c65536}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_15_gpt_16x1024x4096_4096_4096_dispatch_1::@cuda_nvptx_fb::@f_15_gpt_16x1024x4096_4096_4096_dispatch_1_generic_16x1024x4096_f32 {
        ro %arg3[%c0 for %c268435456] : !stream.resource<external>{%c268435456},
        ro %arg7[%c0 for %c65536] : !stream.resource<transient>{%c65536},
        ro %arg4[%c0 for %c16384] : !stream.resource<external>{%c16384},
        ro %arg5[%c0 for %c16384] : !stream.resource<external>{%c16384},
        wo %arg6[%c0 for %c268435456] : !stream.resource<external>{%c268435456}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>, #hal.interface.binding<0, 4>]}
    } => !stream.timepoint
    %5 = stream.resource.dealloca await(%4) => %result : !stream.resource<transient>{%c65536} => !stream.timepoint
    %6 = stream.timepoint.await %5 => %3 : !stream.resource<external>{%c268435456}
    %7 = stream.tensor.export %6 : tensor<16x1024x4096xf32> in !stream.resource<external>{%c268435456} -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
}
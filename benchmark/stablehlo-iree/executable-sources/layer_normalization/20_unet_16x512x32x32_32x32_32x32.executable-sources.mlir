module @f_20_unet_16x512x32x32_32x32_32x32 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_20_unet_16x512x32x32_32x32_32x32_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_20_unet_16x512x32x32_32x32_32x32_dispatch_0_generic_8192x1024_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_20_unet_16x512x32x32_32x32_32x32_dispatch_0_generic_8192x1024_f32() {
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = arith.index_castui %0 {stream.alignment = 32768 : index, stream.values = [0 : index, 32768 : index]} : i32 to index
          %3 = arith.index_castui %1 {stream.alignment = 32768 : index, stream.values = [0 : index, 33587200 : index]} : i32 to index
          %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%2) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x1024xf32>>
          %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%3) : !flow.dispatch.tensor<readwrite:tensor<8192xf32>>
          %6 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [8192, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x1024xf32>> -> tensor<8192x1024xf32>
          %7 = flow.dispatch.tensor.load %5, offsets = [0], sizes = [8192], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<8192xf32>> -> tensor<8192xf32>
          %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%6 : tensor<8192x1024xf32>) outs(%7 : tensor<8192xf32>) {
          ^bb0(%in: f32, %out: f32):
            %9 = arith.addf %out, %in : f32
            linalg.yield %9 : f32
          } -> tensor<8192xf32>
          flow.dispatch.tensor.store %8, %5, offsets = [0], sizes = [8192], strides = [1] : tensor<8192xf32> -> !flow.dispatch.tensor<readwrite:tensor<8192xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_20_unet_16x512x32x32_32x32_32x32_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_20_unet_16x512x32x32_32x32_32x32_dispatch_1_generic_8388608_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_20_unet_16x512x32x32_32x32_32x32_dispatch_1_generic_8388608_f32() {
          %c0 = arith.constant 0 : index
          %c32768 = arith.constant 32768 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8388608xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c32768) : !flow.dispatch.tensor<writeonly:tensor<8388608xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [8388608], strides = [1] : !flow.dispatch.tensor<readonly:tensor<8388608xf32>> -> tensor<8388608xf32>
          %3 = tensor.empty() : tensor<8388608xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2 : tensor<8388608xf32>) outs(%3 : tensor<8388608xf32>) {
          ^bb0(%in: f32, %out: f32):
            %5 = arith.mulf %in, %in : f32
            linalg.yield %5 : f32
          } -> tensor<8388608xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0], sizes = [8388608], strides = [1] : tensor<8388608xf32> -> !flow.dispatch.tensor<writeonly:tensor<8388608xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_20_unet_16x512x32x32_32x32_32x32_dispatch_3 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_20_unet_16x512x32x32_32x32_32x32_dispatch_3_generic_16x512x32x32_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer, ReadOnly>, <4, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_20_unet_16x512x32x32_32x32_32x32_dispatch_3_generic_16x512x32x32_f32() {
          %c0 = arith.constant 0 : index
          %c33587200 = arith.constant 33587200 : index
          %cst = arith.constant 1.024000e+03 : f32
          %cst_0 = arith.constant 9.99999974E-6 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x512x32x32xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x512xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c33587200) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x512xf32>>
          %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x32xf32>>
          %4 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x32xf32>>
          %5 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x512x32x32xf32>>
          %6 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [16, 512, 32, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x512x32x32xf32>> -> tensor<16x512x32x32xf32>
          %7 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x512xf32>> -> tensor<16x512xf32>
          %8 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [16, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x512xf32>> -> tensor<16x512xf32>
          %9 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [32, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x32xf32>> -> tensor<32x32xf32>
          %10 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [32, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x32xf32>> -> tensor<32x32xf32>
          %11 = tensor.empty() : tensor<16x512x32x32xf32>
          %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6, %7, %8, %9, %10 : tensor<16x512x32x32xf32>, tensor<16x512xf32>, tensor<16x512xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%11 : tensor<16x512x32x32xf32>) {
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
          } -> tensor<16x512x32x32xf32>
          flow.dispatch.tensor.store %12, %5, offsets = [0, 0, 0, 0], sizes = [16, 512, 32, 32], strides = [1, 1, 1, 1] : tensor<16x512x32x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x512x32x32xf32>>
          return
        }
      }
    }
  }
  func.func @f_20_unet_16x512x32x32_32x32_32x32(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c33587200_i32 = arith.constant 33587200 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %c0_i32 = arith.constant 0 : i32
    %c33619968 = arith.constant 33619968 : index
    %c33587200 = arith.constant 33587200 : index
    %c33554432 = arith.constant 33554432 : index
    %c4096 = arith.constant 4096 : index
    %c32768 = arith.constant 32768 : index
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index
    %c32 = arith.constant 32 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c16, %c512, %c32, %c32]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<16x512x32x32xf32> in !stream.resource<external>{%c33554432}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c32, %c32]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<32x32xf32> in !stream.resource<external>{%c4096}
    hal.buffer_view.assert<%arg2 : !hal.buffer_view> message("input 2") shape([%c32, %c32]) type(%c553648160_i32) encoding(%c1_i32)
    %2 = stream.tensor.import %arg2 : !hal.buffer_view -> tensor<32x32xf32> in !stream.resource<external>{%c4096}
    %3 = stream.resource.alloc uninitialized : !stream.resource<external>{%c33554432}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c33619968} => !stream.timepoint
    %4 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg3: !stream.resource<external>{%c33554432}, %1 as %arg4: !stream.resource<external>{%c4096}, %2 as %arg5: !stream.resource<external>{%c4096}, %3 as %arg6: !stream.resource<external>{%c33554432}, %result as %arg7: !stream.resource<transient>{%c33619968}) {
      stream.cmd.fill %c0_i8, %arg7[%c0 for %c32768] : i8 -> !stream.resource<transient>{%c33619968}
      stream.cmd.concurrent {
        stream.cmd.dispatch @f_20_unet_16x512x32x32_32x32_32x32_dispatch_0::@cuda_nvptx_fb::@f_20_unet_16x512x32x32_32x32_32x32_dispatch_0_generic_8192x1024_f32(%c0_i32, %c0_i32 : i32, i32) {
          ro %arg3[%c0 for %c33554432] : !stream.resource<external>{%c33554432},
          rw %arg7[%c0 for %c33619968] : !stream.resource<transient>{%c33619968}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.dispatch @f_20_unet_16x512x32x32_32x32_32x32_dispatch_1::@cuda_nvptx_fb::@f_20_unet_16x512x32x32_32x32_32x32_dispatch_1_generic_8388608_f32 {
          ro %arg3[%c0 for %c33554432] : !stream.resource<external>{%c33554432},
          wo %arg7[%c0 for %c33619968] : !stream.resource<transient>{%c33619968}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.fill %c0_i8, %arg7[%c33587200 for %c32768] : i8 -> !stream.resource<transient>{%c33619968}
      }
      stream.cmd.dispatch @f_20_unet_16x512x32x32_32x32_32x32_dispatch_0::@cuda_nvptx_fb::@f_20_unet_16x512x32x32_32x32_32x32_dispatch_0_generic_8192x1024_f32(%c32768_i32, %c33587200_i32 : i32, i32) {
        ro %arg7[%c0 for %c33619968] : !stream.resource<transient>{%c33619968},
        rw %arg7[%c0 for %c33619968] : !stream.resource<transient>{%c33619968}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_20_unet_16x512x32x32_32x32_32x32_dispatch_3::@cuda_nvptx_fb::@f_20_unet_16x512x32x32_32x32_32x32_dispatch_3_generic_16x512x32x32_f32 {
        ro %arg3[%c0 for %c33554432] : !stream.resource<external>{%c33554432},
        ro %arg7[%c0 for %c33619968] : !stream.resource<transient>{%c33619968},
        ro %arg4[%c0 for %c4096] : !stream.resource<external>{%c4096},
        ro %arg5[%c0 for %c4096] : !stream.resource<external>{%c4096},
        wo %arg6[%c0 for %c33554432] : !stream.resource<external>{%c33554432}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>, #hal.interface.binding<0, 4>]}
    } => !stream.timepoint
    %5 = stream.resource.dealloca await(%4) => %result : !stream.resource<transient>{%c33619968} => !stream.timepoint
    %6 = stream.timepoint.await %5 => %3 : !stream.resource<external>{%c33554432}
    %7 = stream.tensor.export %6 : tensor<16x512x32x32xf32> in !stream.resource<external>{%c33554432} -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
}
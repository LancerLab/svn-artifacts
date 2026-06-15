module @f_14_efficientnet_64x1280x7x7_7x7_7x7 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_14_efficientnet_64x1280x7x7_7x7_7x7_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_14_efficientnet_64x1280x7x7_7x7_7x7_dispatch_0_generic_81920x49_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_14_efficientnet_64x1280x7x7_7x7_7x7_dispatch_0_generic_81920x49_f32() {
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = arith.index_castui %0 {stream.alignment = 65536 : index, stream.values = [0 : index, 327680 : index]} : i32 to index
          %3 = arith.index_castui %1 {stream.alignment = 131072 : index, stream.values = [0 : index, 16384000 : index]} : i32 to index
          %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%2) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<81920x49xf32>>
          %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%3) : !flow.dispatch.tensor<readwrite:tensor<81920xf32>>
          %6 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [81920, 49], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<81920x49xf32>> -> tensor<81920x49xf32>
          %7 = flow.dispatch.tensor.load %5, offsets = [0], sizes = [81920], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<81920xf32>> -> tensor<81920xf32>
          %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%6 : tensor<81920x49xf32>) outs(%7 : tensor<81920xf32>) {
          ^bb0(%in: f32, %out: f32):
            %9 = arith.addf %out, %in : f32
            linalg.yield %9 : f32
          } -> tensor<81920xf32>
          flow.dispatch.tensor.store %8, %5, offsets = [0], sizes = [81920], strides = [1] : tensor<81920xf32> -> !flow.dispatch.tensor<readwrite:tensor<81920xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_14_efficientnet_64x1280x7x7_7x7_7x7_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_14_efficientnet_64x1280x7x7_7x7_7x7_dispatch_1_generic_4014080_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_14_efficientnet_64x1280x7x7_7x7_7x7_dispatch_1_generic_4014080_f32() {
          %c0 = arith.constant 0 : index
          %c327680 = arith.constant 327680 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4014080xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c327680) : !flow.dispatch.tensor<writeonly:tensor<4014080xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [4014080], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4014080xf32>> -> tensor<4014080xf32>
          %3 = tensor.empty() : tensor<4014080xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2 : tensor<4014080xf32>) outs(%3 : tensor<4014080xf32>) {
          ^bb0(%in: f32, %out: f32):
            %5 = arith.mulf %in, %in : f32
            linalg.yield %5 : f32
          } -> tensor<4014080xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0], sizes = [4014080], strides = [1] : tensor<4014080xf32> -> !flow.dispatch.tensor<writeonly:tensor<4014080xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_14_efficientnet_64x1280x7x7_7x7_7x7_dispatch_3 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_14_efficientnet_64x1280x7x7_7x7_7x7_dispatch_3_generic_64x1280x7x7_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer, ReadOnly>, <4, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_14_efficientnet_64x1280x7x7_7x7_7x7_dispatch_3_generic_64x1280x7x7_f32() {
          %c0 = arith.constant 0 : index
          %c16384000 = arith.constant 16384000 : index
          %cst = arith.constant 4.900000e+01 : f32
          %cst_0 = arith.constant 9.99999974E-6 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x1280x7x7xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x1280xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c16384000) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x1280xf32>>
          %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<7x7xf32>>
          %4 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<7x7xf32>>
          %5 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x1280x7x7xf32>>
          %6 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [64, 1280, 7, 7], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x1280x7x7xf32>> -> tensor<64x1280x7x7xf32>
          %7 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [64, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x1280xf32>> -> tensor<64x1280xf32>
          %8 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [64, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x1280xf32>> -> tensor<64x1280xf32>
          %9 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [7, 7], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<7x7xf32>> -> tensor<7x7xf32>
          %10 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [7, 7], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<7x7xf32>> -> tensor<7x7xf32>
          %11 = tensor.empty() : tensor<64x1280x7x7xf32>
          %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6, %7, %8, %9, %10 : tensor<64x1280x7x7xf32>, tensor<64x1280xf32>, tensor<64x1280xf32>, tensor<7x7xf32>, tensor<7x7xf32>) outs(%11 : tensor<64x1280x7x7xf32>) {
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
          } -> tensor<64x1280x7x7xf32>
          flow.dispatch.tensor.store %12, %5, offsets = [0, 0, 0, 0], sizes = [64, 1280, 7, 7], strides = [1, 1, 1, 1] : tensor<64x1280x7x7xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x1280x7x7xf32>>
          return
        }
      }
    }
  }
  func.func @f_14_efficientnet_64x1280x7x7_7x7_7x7(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c16384000_i32 = arith.constant 16384000 : i32
    %c327680_i32 = arith.constant 327680 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16711680 = arith.constant 16711680 : index
    %c16384000 = arith.constant 16384000 : index
    %c16056320 = arith.constant 16056320 : index
    %c196 = arith.constant 196 : index
    %c327680 = arith.constant 327680 : index
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c1280 = arith.constant 1280 : index
    %c7 = arith.constant 7 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c64, %c1280, %c7, %c7]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<64x1280x7x7xf32> in !stream.resource<external>{%c16056320}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c7, %c7]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<7x7xf32> in !stream.resource<external>{%c196}
    hal.buffer_view.assert<%arg2 : !hal.buffer_view> message("input 2") shape([%c7, %c7]) type(%c553648160_i32) encoding(%c1_i32)
    %2 = stream.tensor.import %arg2 : !hal.buffer_view -> tensor<7x7xf32> in !stream.resource<external>{%c196}
    %3 = stream.resource.alloc uninitialized : !stream.resource<external>{%c16056320}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c16711680} => !stream.timepoint
    %4 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg3: !stream.resource<external>{%c16056320}, %1 as %arg4: !stream.resource<external>{%c196}, %2 as %arg5: !stream.resource<external>{%c196}, %3 as %arg6: !stream.resource<external>{%c16056320}, %result as %arg7: !stream.resource<transient>{%c16711680}) {
      stream.cmd.fill %c0_i8, %arg7[%c0 for %c327680] : i8 -> !stream.resource<transient>{%c16711680}
      stream.cmd.concurrent {
        stream.cmd.dispatch @f_14_efficientnet_64x1280x7x7_7x7_7x7_dispatch_0::@cuda_nvptx_fb::@f_14_efficientnet_64x1280x7x7_7x7_7x7_dispatch_0_generic_81920x49_f32(%c0_i32, %c0_i32 : i32, i32) {
          ro %arg3[%c0 for %c16056320] : !stream.resource<external>{%c16056320},
          rw %arg7[%c0 for %c16711680] : !stream.resource<transient>{%c16711680}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.dispatch @f_14_efficientnet_64x1280x7x7_7x7_7x7_dispatch_1::@cuda_nvptx_fb::@f_14_efficientnet_64x1280x7x7_7x7_7x7_dispatch_1_generic_4014080_f32 {
          ro %arg3[%c0 for %c16056320] : !stream.resource<external>{%c16056320},
          wo %arg7[%c0 for %c16711680] : !stream.resource<transient>{%c16711680}
        } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
        stream.cmd.fill %c0_i8, %arg7[%c16384000 for %c327680] : i8 -> !stream.resource<transient>{%c16711680}
      }
      stream.cmd.dispatch @f_14_efficientnet_64x1280x7x7_7x7_7x7_dispatch_0::@cuda_nvptx_fb::@f_14_efficientnet_64x1280x7x7_7x7_7x7_dispatch_0_generic_81920x49_f32(%c327680_i32, %c16384000_i32 : i32, i32) {
        ro %arg7[%c0 for %c16711680] : !stream.resource<transient>{%c16711680},
        rw %arg7[%c0 for %c16711680] : !stream.resource<transient>{%c16711680}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_14_efficientnet_64x1280x7x7_7x7_7x7_dispatch_3::@cuda_nvptx_fb::@f_14_efficientnet_64x1280x7x7_7x7_7x7_dispatch_3_generic_64x1280x7x7_f32 {
        ro %arg3[%c0 for %c16056320] : !stream.resource<external>{%c16056320},
        ro %arg7[%c0 for %c16711680] : !stream.resource<transient>{%c16711680},
        ro %arg4[%c0 for %c196] : !stream.resource<external>{%c196},
        ro %arg5[%c0 for %c196] : !stream.resource<external>{%c196},
        wo %arg6[%c0 for %c16056320] : !stream.resource<external>{%c16056320}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>, #hal.interface.binding<0, 4>]}
    } => !stream.timepoint
    %5 = stream.resource.dealloca await(%4) => %result : !stream.resource<transient>{%c16711680} => !stream.timepoint
    %6 = stream.timepoint.await %5 => %3 : !stream.resource<external>{%c16056320}
    %7 = stream.tensor.export %6 : tensor<64x1280x7x7xf32> in !stream.resource<external>{%c16056320} -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
}
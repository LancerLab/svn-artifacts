module @f_14_efficientnet_64x1280x7x7_64x1280x7x7 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_14_efficientnet_64x1280x7x7_64x1280x7x7_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_14_efficientnet_64x1280x7x7_64x1280x7x7_dispatch_0_generic_64x1280x7x7_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_14_efficientnet_64x1280x7x7_64x1280x7x7_dispatch_0_generic_64x1280x7x7_f32() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant -3.402820e+38 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x1280x7x7xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x1280x7x7xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [64, 1280, 7, 7], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x1280x7x7xf32>> -> tensor<64x1280x7x7xf32>
          %3 = tensor.empty() : tensor<64x1280x7x7xf32>
          %4 = tensor.empty() : tensor<64x1280x7xf32>
          %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<64x1280x7xf32>) -> tensor<64x1280x7xf32>
          %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2 : tensor<64x1280x7x7xf32>) outs(%5 : tensor<64x1280x7xf32>) {
          ^bb0(%in: f32, %out: f32):
            %8 = arith.maxf %out, %in : f32
            linalg.yield %8 : f32
          } -> tensor<64x1280x7xf32>
          %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %6 : tensor<64x1280x7x7xf32>, tensor<64x1280x7xf32>) outs(%3 : tensor<64x1280x7x7xf32>) {
          ^bb0(%in: f32, %in_0: f32, %out: f32):
            %8 = arith.subf %in, %in_0 : f32
            %9 = math.exp %8 : f32
            linalg.yield %9 : f32
          } -> tensor<64x1280x7x7xf32>
          flow.dispatch.tensor.store %7, %1, offsets = [0, 0, 0, 0], sizes = [64, 1280, 7, 7], strides = [1, 1, 1, 1] : tensor<64x1280x7x7xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x1280x7x7xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_14_efficientnet_64x1280x7x7_64x1280x7x7_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_14_efficientnet_64x1280x7x7_64x1280x7x7_dispatch_1_generic_64x1280x7x7_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_14_efficientnet_64x1280x7x7_64x1280x7x7_dispatch_1_generic_64x1280x7x7_f32() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x1280x7x7xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x1280x7x7xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [64, 1280, 7, 7], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x1280x7x7xf32>> -> tensor<64x1280x7x7xf32>
          %3 = tensor.empty() : tensor<64x1280x7x7xf32>
          %4 = tensor.empty() : tensor<64x1280x7xf32>
          %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<64x1280x7xf32>) -> tensor<64x1280x7xf32>
          %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2 : tensor<64x1280x7x7xf32>) outs(%5 : tensor<64x1280x7xf32>) {
          ^bb0(%in: f32, %out: f32):
            %8 = arith.addf %out, %in : f32
            linalg.yield %8 : f32
          } -> tensor<64x1280x7xf32>
          %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %6 : tensor<64x1280x7x7xf32>, tensor<64x1280x7xf32>) outs(%3 : tensor<64x1280x7x7xf32>) {
          ^bb0(%in: f32, %in_0: f32, %out: f32):
            %8 = arith.divf %in, %in_0 : f32
            linalg.yield %8 : f32
          } -> tensor<64x1280x7x7xf32>
          flow.dispatch.tensor.store %7, %1, offsets = [0, 0, 0, 0], sizes = [64, 1280, 7, 7], strides = [1, 1, 1, 1] : tensor<64x1280x7x7xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x1280x7x7xf32>>
          return
        }
      }
    }
  }
  func.func @f_14_efficientnet_64x1280x7x7_64x1280x7x7(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c16056320 = arith.constant 16056320 : index
    %c0 = arith.constant 0 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c1280 = arith.constant 1280 : index
    %c7 = arith.constant 7 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c64, %c1280, %c7, %c7]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<64x1280x7x7xf32> in !stream.resource<external>{%c16056320}
    %1 = stream.resource.alloc uninitialized : !stream.resource<external>{%c16056320}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c16056320} => !stream.timepoint
    %2 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg1: !stream.resource<external>{%c16056320}, %1 as %arg2: !stream.resource<external>{%c16056320}, %result as %arg3: !stream.resource<transient>{%c16056320}) {
      stream.cmd.dispatch @f_14_efficientnet_64x1280x7x7_64x1280x7x7_dispatch_0::@cuda_nvptx_fb::@f_14_efficientnet_64x1280x7x7_64x1280x7x7_dispatch_0_generic_64x1280x7x7_f32 {
        ro %arg1[%c0 for %c16056320] : !stream.resource<external>{%c16056320},
        wo %arg3[%c0 for %c16056320] : !stream.resource<transient>{%c16056320}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_14_efficientnet_64x1280x7x7_64x1280x7x7_dispatch_1::@cuda_nvptx_fb::@f_14_efficientnet_64x1280x7x7_64x1280x7x7_dispatch_1_generic_64x1280x7x7_f32 {
        ro %arg3[%c0 for %c16056320] : !stream.resource<transient>{%c16056320},
        wo %arg2[%c0 for %c16056320] : !stream.resource<external>{%c16056320}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %3 = stream.resource.dealloca await(%2) => %result : !stream.resource<transient>{%c16056320} => !stream.timepoint
    %4 = stream.timepoint.await %3 => %1 : !stream.resource<external>{%c16056320}
    %5 = stream.tensor.export %4 : tensor<64x1280x7x7xf32> in !stream.resource<external>{%c16056320} -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
module @f_13_efficientnet_64x1280x7x7_64x1280 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_13_efficientnet_64x1280x7x7_64x1280_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_13_efficientnet_64x1280x7x7_64x1280_dispatch_0_generic_64x1280x49_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_13_efficientnet_64x1280x7x7_64x1280_dispatch_0_generic_64x1280x49_f32() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %cst_0 = arith.constant 4.900000e+01 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x1280x49xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x1280xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 1280, 49], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x1280x49xf32>> -> tensor<64x1280x49xf32>
          %3 = tensor.empty() : tensor<64x1280xf32>
          %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<64x1280xf32>) -> tensor<64x1280xf32>
          %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2 : tensor<64x1280x49xf32>) outs(%4 : tensor<64x1280xf32>) {
          ^bb0(%in: f32, %out: f32):
            %7 = arith.addf %out, %in : f32
            linalg.yield %7 : f32
          } -> tensor<64x1280xf32>
          %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<64x1280xf32>) outs(%3 : tensor<64x1280xf32>) {
          ^bb0(%in: f32, %out: f32):
            %7 = arith.divf %in, %cst_0 : f32
            linalg.yield %7 : f32
          } -> tensor<64x1280xf32>
          flow.dispatch.tensor.store %6, %1, offsets = [0, 0], sizes = [64, 1280], strides = [1, 1] : tensor<64x1280xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x1280xf32>>
          return
        }
      }
    }
  }
  func.func @f_13_efficientnet_64x1280x7x7_64x1280(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c16056320 = arith.constant 16056320 : index
    %c327680 = arith.constant 327680 : index
    %c0 = arith.constant 0 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c1280 = arith.constant 1280 : index
    %c7 = arith.constant 7 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c64, %c1280, %c7, %c7]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<64x1280x7x7xf32> in !stream.resource<external>{%c16056320}
    %1 = stream.resource.alloc uninitialized : !stream.resource<external>{%c327680}
    %2 = stream.cmd.execute with(%0 as %arg1: !stream.resource<external>{%c16056320}, %1 as %arg2: !stream.resource<external>{%c327680}) {
      stream.cmd.dispatch @f_13_efficientnet_64x1280x7x7_64x1280_dispatch_0::@cuda_nvptx_fb::@f_13_efficientnet_64x1280x7x7_64x1280_dispatch_0_generic_64x1280x49_f32 {
        ro %arg1[%c0 for %c16056320] : !stream.resource<external>{%c16056320},
        wo %arg2[%c0 for %c327680] : !stream.resource<external>{%c327680}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %3 = stream.timepoint.await %2 => %1 : !stream.resource<external>{%c327680}
    %4 = stream.tensor.export %3 : tensor<64x1280xf32> in !stream.resource<external>{%c327680} -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
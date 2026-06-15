module @f_16_mobilenet_128x96x112x112_128x112x112 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_16_mobilenet_128x96x112x112_128x112x112_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_16_mobilenet_128x96x112x112_128x112x112_dispatch_0_generic_128x112x112x96_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_16_mobilenet_128x96x112x112_128x112x112_dispatch_0_generic_128x112x112x96_f32() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %cst_0 = arith.constant 9.600000e+01 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x96x112x112xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x112x112xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [128, 96, 112, 112], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x96x112x112xf32>> -> tensor<128x96x112x112xf32>
          %3 = tensor.empty() : tensor<128x112x112xf32>
          %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<128x112x112xf32>) -> tensor<128x112x112xf32>
          %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2 : tensor<128x96x112x112xf32>) outs(%4 : tensor<128x112x112xf32>) {
          ^bb0(%in: f32, %out: f32):
            %7 = arith.addf %out, %in : f32
            linalg.yield %7 : f32
          } -> tensor<128x112x112xf32>
          %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<128x112x112xf32>) outs(%3 : tensor<128x112x112xf32>) {
          ^bb0(%in: f32, %out: f32):
            %7 = arith.divf %in, %cst_0 : f32
            linalg.yield %7 : f32
          } -> tensor<128x112x112xf32>
          flow.dispatch.tensor.store %6, %1, offsets = [0, 0, 0], sizes = [128, 112, 112], strides = [1, 1, 1] : tensor<128x112x112xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x112x112xf32>>
          return
        }
      }
    }
  }
  func.func @f_16_mobilenet_128x96x112x112_128x112x112(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c616562688 = arith.constant 616562688 : index
    %c6422528 = arith.constant 6422528 : index
    %c0 = arith.constant 0 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128 = arith.constant 128 : index
    %c96 = arith.constant 96 : index
    %c112 = arith.constant 112 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c128, %c96, %c112, %c112]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<128x96x112x112xf32> in !stream.resource<external>{%c616562688}
    %1 = stream.resource.alloc uninitialized : !stream.resource<external>{%c6422528}
    %2 = stream.cmd.execute with(%0 as %arg1: !stream.resource<external>{%c616562688}, %1 as %arg2: !stream.resource<external>{%c6422528}) {
      stream.cmd.dispatch @f_16_mobilenet_128x96x112x112_128x112x112_dispatch_0::@cuda_nvptx_fb::@f_16_mobilenet_128x96x112x112_128x112x112_dispatch_0_generic_128x112x112x96_f32 {
        ro %arg1[%c0 for %c616562688] : !stream.resource<external>{%c616562688},
        wo %arg2[%c0 for %c6422528] : !stream.resource<external>{%c6422528}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %3 = stream.timepoint.await %2 => %1 : !stream.resource<external>{%c6422528}
    %4 = stream.tensor.export %3 : tensor<128x112x112xf32> in !stream.resource<external>{%c6422528} -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
module @f_20_vit_32x197x768_32x768 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_20_vit_32x197x768_32x768_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_20_vit_32x197x768_32x768_dispatch_0_generic_32x768x197_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_20_vit_32x197x768_32x768_dispatch_0_generic_32x768x197_f32() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %cst_0 = arith.constant 1.970000e+02 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x197x768xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x768xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [32, 197, 768], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x197x768xf32>> -> tensor<32x197x768xf32>
          %3 = tensor.empty() : tensor<32x768xf32>
          %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<32x768xf32>) -> tensor<32x768xf32>
          %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2 : tensor<32x197x768xf32>) outs(%4 : tensor<32x768xf32>) {
          ^bb0(%in: f32, %out: f32):
            %7 = arith.addf %out, %in : f32
            linalg.yield %7 : f32
          } -> tensor<32x768xf32>
          %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<32x768xf32>) outs(%3 : tensor<32x768xf32>) {
          ^bb0(%in: f32, %out: f32):
            %7 = arith.divf %in, %cst_0 : f32
            linalg.yield %7 : f32
          } -> tensor<32x768xf32>
          flow.dispatch.tensor.store %6, %1, offsets = [0, 0], sizes = [32, 768], strides = [1, 1] : tensor<32x768xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x768xf32>>
          return
        }
      }
    }
  }
  func.func @f_20_vit_32x197x768_32x768(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c19365888 = arith.constant 19365888 : index
    %c98304 = arith.constant 98304 : index
    %c0 = arith.constant 0 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32 = arith.constant 32 : index
    %c197 = arith.constant 197 : index
    %c768 = arith.constant 768 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c32, %c197, %c768]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<32x197x768xf32> in !stream.resource<external>{%c19365888}
    %1 = stream.resource.alloc uninitialized : !stream.resource<external>{%c98304}
    %2 = stream.cmd.execute with(%0 as %arg1: !stream.resource<external>{%c19365888}, %1 as %arg2: !stream.resource<external>{%c98304}) {
      stream.cmd.dispatch @f_20_vit_32x197x768_32x768_dispatch_0::@cuda_nvptx_fb::@f_20_vit_32x197x768_32x768_dispatch_0_generic_32x768x197_f32 {
        ro %arg1[%c0 for %c19365888] : !stream.resource<external>{%c19365888},
        wo %arg2[%c0 for %c98304] : !stream.resource<external>{%c98304}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %3 = stream.timepoint.await %2 => %1 : !stream.resource<external>{%c98304}
    %4 = stream.tensor.export %3 : tensor<32x768xf32> in !stream.resource<external>{%c98304} -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
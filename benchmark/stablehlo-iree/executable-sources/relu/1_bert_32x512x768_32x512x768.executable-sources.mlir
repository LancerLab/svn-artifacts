module @f_1_bert_32x512x768_32x512x768 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_1_bert_32x512x768_32x512x768_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_1_bert_32x512x768_32x512x768_dispatch_0_generic_12582912_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_1_bert_32x512x768_32x512x768_dispatch_0_generic_12582912_f32() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %cst_0 = arith.constant 3.40282347E+38 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<12582912xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<12582912xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [12582912], strides = [1] : !flow.dispatch.tensor<readonly:tensor<12582912xf32>> -> tensor<12582912xf32>
          %3 = tensor.empty() : tensor<12582912xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2 : tensor<12582912xf32>) outs(%3 : tensor<12582912xf32>) {
          ^bb0(%in: f32, %out: f32):
            %5 = arith.maxf %in, %cst : f32
            %6 = arith.minf %5, %cst_0 : f32
            linalg.yield %6 : f32
          } -> tensor<12582912xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0], sizes = [12582912], strides = [1] : tensor<12582912xf32> -> !flow.dispatch.tensor<writeonly:tensor<12582912xf32>>
          return
        }
      }
    }
  }
  func.func @f_1_bert_32x512x768_32x512x768(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c50331648 = arith.constant 50331648 : index
    %c0 = arith.constant 0 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c768 = arith.constant 768 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c32, %c512, %c768]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<32x512x768xf32> in !stream.resource<external>{%c50331648}
    %1 = stream.resource.alloc uninitialized : !stream.resource<external>{%c50331648}
    %2 = stream.cmd.execute with(%0 as %arg1: !stream.resource<external>{%c50331648}, %1 as %arg2: !stream.resource<external>{%c50331648}) {
      stream.cmd.dispatch @f_1_bert_32x512x768_32x512x768_dispatch_0::@cuda_nvptx_fb::@f_1_bert_32x512x768_32x512x768_dispatch_0_generic_12582912_f32 {
        ro %arg1[%c0 for %c50331648] : !stream.resource<external>{%c50331648},
        wo %arg2[%c0 for %c50331648] : !stream.resource<external>{%c50331648}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %3 = stream.timepoint.await %2 => %1 : !stream.resource<external>{%c50331648}
    %4 = stream.tensor.export %3 : tensor<32x512x768xf32> in !stream.resource<external>{%c50331648} -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
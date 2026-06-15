module @f_15_gpt_16x1024x4096 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_15_gpt_16x1024x4096_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_15_gpt_16x1024x4096_dispatch_0_generic_67108864_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_15_gpt_16x1024x4096_dispatch_0_generic_67108864_f32() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 1.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<67108864xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<67108864xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [67108864], strides = [1] : !flow.dispatch.tensor<readonly:tensor<67108864xf32>> -> tensor<67108864xf32>
          %3 = tensor.empty() : tensor<67108864xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2 : tensor<67108864xf32>) outs(%3 : tensor<67108864xf32>) {
          ^bb0(%in: f32, %out: f32):
            %5 = arith.negf %in : f32
            %6 = math.exp %5 : f32
            %7 = arith.addf %6, %cst : f32
            %8 = arith.divf %cst, %7 : f32
            linalg.yield %8 : f32
          } -> tensor<67108864xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0], sizes = [67108864], strides = [1] : tensor<67108864xf32> -> !flow.dispatch.tensor<writeonly:tensor<67108864xf32>>
          return
        }
      }
    }
  }
  func.func @f_15_gpt_16x1024x4096(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c268435456 = arith.constant 268435456 : index
    %c0 = arith.constant 0 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %c1024 = arith.constant 1024 : index
    %c4096 = arith.constant 4096 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c16, %c1024, %c4096]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<16x1024x4096xf32> in !stream.resource<external>{%c268435456}
    %1 = stream.resource.alloc uninitialized : !stream.resource<external>{%c268435456}
    %2 = stream.cmd.execute with(%0 as %arg1: !stream.resource<external>{%c268435456}, %1 as %arg2: !stream.resource<external>{%c268435456}) {
      stream.cmd.dispatch @f_15_gpt_16x1024x4096_dispatch_0::@cuda_nvptx_fb::@f_15_gpt_16x1024x4096_dispatch_0_generic_67108864_f32 {
        ro %arg1[%c0 for %c268435456] : !stream.resource<external>{%c268435456},
        wo %arg2[%c0 for %c268435456] : !stream.resource<external>{%c268435456}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %3 = stream.timepoint.await %2 => %1 : !stream.resource<external>{%c268435456}
    %4 = stream.tensor.export %3 : tensor<16x1024x4096xf32> in !stream.resource<external>{%c268435456} -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
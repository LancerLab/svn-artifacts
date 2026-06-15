module @f_14_gpt_16x1024_50257x1024_16x1024x1024 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_14_gpt_16x1024_50257x1024_16x1024x1024_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_14_gpt_16x1024_50257x1024_16x1024x1024_dispatch_0_generic_16x1024x1024_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_14_gpt_16x1024_50257x1024_16x1024x1024_dispatch_0_generic_16x1024x1024_f32() {
          %c0 = arith.constant 0 : index
          %c50256 = arith.constant 50256 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x1024xi32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<50257x1024xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x1024x1024xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x1024xi32>> -> tensor<16x1024xi32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [50257, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<50257x1024xf32>> -> tensor<50257x1024xf32>
          %5 = tensor.empty() : tensor<16x1024x1024xf32>
          %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} outs(%5 : tensor<16x1024x1024xf32>) {
          ^bb0(%out: f32):
            %7 = linalg.index 0 : index
            %8 = linalg.index 1 : index
            %9 = linalg.index 2 : index
            %extracted = tensor.extract %3[%7, %8] : tensor<16x1024xi32>
            %10 = arith.index_cast %extracted : i32 to index
            %11 = arith.maxsi %10, %c0 : index
            %12 = arith.minsi %11, %c50256 : index
            %extracted_0 = tensor.extract %4[%12, %9] : tensor<50257x1024xf32>
            linalg.yield %extracted_0 : f32
          } -> tensor<16x1024x1024xf32>
          flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [16, 1024, 1024], strides = [1, 1, 1] : tensor<16x1024x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x1024x1024xf32>>
          return
        }
      }
    }
  }
  func.func @f_14_gpt_16x1024_50257x1024_16x1024x1024(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c205852672 = arith.constant 205852672 : index
    %c65536 = arith.constant 65536 : index
    %c67108864 = arith.constant 67108864 : index
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c268435488_i32 = arith.constant 268435488 : i32
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c50257 = arith.constant 50257 : index
    %c1024 = arith.constant 1024 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c50257, %c1024]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<50257x1024xf32> in !stream.resource<external>{%c205852672}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c16, %c1024]) type(%c268435488_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<16x1024xi32> in !stream.resource<external>{%c65536}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c67108864}
    %3 = stream.cmd.execute with(%1 as %arg2: !stream.resource<external>{%c65536}, %0 as %arg3: !stream.resource<external>{%c205852672}, %2 as %arg4: !stream.resource<external>{%c67108864}) {
      stream.cmd.dispatch @f_14_gpt_16x1024_50257x1024_16x1024x1024_dispatch_0::@cuda_nvptx_fb::@f_14_gpt_16x1024_50257x1024_16x1024x1024_dispatch_0_generic_16x1024x1024_f32 {
        ro %arg2[%c0 for %c65536] : !stream.resource<external>{%c65536},
        ro %arg3[%c0 for %c205852672] : !stream.resource<external>{%c205852672},
        wo %arg4[%c0 for %c67108864] : !stream.resource<external>{%c67108864}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]}
    } => !stream.timepoint
    %4 = stream.timepoint.await %3 => %2 : !stream.resource<external>{%c67108864}
    %5 = stream.tensor.export %4 : tensor<16x1024x1024xf32> in !stream.resource<external>{%c67108864} -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
module @f_15_gpt_16x1024x1536_1536x6144_16x1024x6144 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_15_gpt_16x1024x1536_1536x6144_16x1024x6144_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_15_gpt_16x1024x1536_1536x6144_16x1024x6144_dispatch_0_matmul_16384x6144x1536_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_15_gpt_16x1024x1536_1536x6144_16x1024x6144_dispatch_0_matmul_16384x6144x1536_f32() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16384x1536xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1536x6144xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16384x6144xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16384, 1536], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16384x1536xf32>> -> tensor<16384x1536xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1536, 6144], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1536x6144xf32>> -> tensor<1536x6144xf32>
          %5 = tensor.empty() : tensor<16384x6144xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<16384x6144xf32>) -> tensor<16384x6144xf32>
          %7 = linalg.matmul ins(%3, %4 : tensor<16384x1536xf32>, tensor<1536x6144xf32>) outs(%6 : tensor<16384x6144xf32>) -> tensor<16384x6144xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [16384, 6144], strides = [1, 1] : tensor<16384x6144xf32> -> !flow.dispatch.tensor<writeonly:tensor<16384x6144xf32>>
          return
        }
      }
    }
  }
  func.func @f_15_gpt_16x1024x1536_1536x6144_16x1024x6144(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c100663296 = arith.constant 100663296 : index
    %c37748736 = arith.constant 37748736 : index
    %c402653184 = arith.constant 402653184 : index
    %c0 = arith.constant 0 : index
    %c6144 = arith.constant 6144 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %c1024 = arith.constant 1024 : index
    %c1536 = arith.constant 1536 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c16, %c1024, %c1536]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<16x1024x1536xf32> in !stream.resource<external>{%c100663296}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c1536, %c6144]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<1536x6144xf32> in !stream.resource<external>{%c37748736}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c402653184}
    %3 = stream.cmd.execute with(%0 as %arg2: !stream.resource<external>{%c100663296}, %1 as %arg3: !stream.resource<external>{%c37748736}, %2 as %arg4: !stream.resource<external>{%c402653184}) {
      stream.cmd.dispatch @f_15_gpt_16x1024x1536_1536x6144_16x1024x6144_dispatch_0::@cuda_nvptx_fb::@f_15_gpt_16x1024x1536_1536x6144_16x1024x6144_dispatch_0_matmul_16384x6144x1536_f32 {
        ro %arg2[%c0 for %c100663296] : !stream.resource<external>{%c100663296},
        ro %arg3[%c0 for %c37748736] : !stream.resource<external>{%c37748736},
        wo %arg4[%c0 for %c402653184] : !stream.resource<external>{%c402653184}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]}
    } => !stream.timepoint
    %4 = stream.timepoint.await %3 => %2 : !stream.resource<external>{%c402653184}
    %5 = stream.tensor.export %4 : tensor<16x1024x6144xf32> in !stream.resource<external>{%c402653184} -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
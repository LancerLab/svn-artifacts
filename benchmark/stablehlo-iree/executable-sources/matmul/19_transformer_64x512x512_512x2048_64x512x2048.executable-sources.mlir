module @f_19_transformer_64x512x512_512x2048_64x512x2048 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_19_transformer_64x512x512_512x2048_64x512x2048_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_19_transformer_64x512x512_512x2048_64x512x2048_dispatch_0_matmul_32768x2048x512_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_19_transformer_64x512x512_512x2048_64x512x2048_dispatch_0_matmul_32768x2048x512_f32() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32768x512xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512x2048xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32768x2048xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32768, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32768x512xf32>> -> tensor<32768x512xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x2048xf32>> -> tensor<512x2048xf32>
          %5 = tensor.empty() : tensor<32768x2048xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<32768x2048xf32>) -> tensor<32768x2048xf32>
          %7 = linalg.matmul ins(%3, %4 : tensor<32768x512xf32>, tensor<512x2048xf32>) outs(%6 : tensor<32768x2048xf32>) -> tensor<32768x2048xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [32768, 2048], strides = [1, 1] : tensor<32768x2048xf32> -> !flow.dispatch.tensor<writeonly:tensor<32768x2048xf32>>
          return
        }
      }
    }
  }
  func.func @f_19_transformer_64x512x512_512x2048_64x512x2048(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c67108864 = arith.constant 67108864 : index
    %c4194304 = arith.constant 4194304 : index
    %c268435456 = arith.constant 268435456 : index
    %c0 = arith.constant 0 : index
    %c2048 = arith.constant 2048 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c64, %c512, %c512]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<64x512x512xf32> in !stream.resource<external>{%c67108864}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c512, %c2048]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<512x2048xf32> in !stream.resource<external>{%c4194304}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c268435456}
    %3 = stream.cmd.execute with(%0 as %arg2: !stream.resource<external>{%c67108864}, %1 as %arg3: !stream.resource<external>{%c4194304}, %2 as %arg4: !stream.resource<external>{%c268435456}) {
      stream.cmd.dispatch @f_19_transformer_64x512x512_512x2048_64x512x2048_dispatch_0::@cuda_nvptx_fb::@f_19_transformer_64x512x512_512x2048_64x512x2048_dispatch_0_matmul_32768x2048x512_f32 {
        ro %arg2[%c0 for %c67108864] : !stream.resource<external>{%c67108864},
        ro %arg3[%c0 for %c4194304] : !stream.resource<external>{%c4194304},
        wo %arg4[%c0 for %c268435456] : !stream.resource<external>{%c268435456}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]}
    } => !stream.timepoint
    %4 = stream.timepoint.await %3 => %2 : !stream.resource<external>{%c268435456}
    %5 = stream.tensor.export %4 : tensor<64x512x2048xf32> in !stream.resource<external>{%c268435456} -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
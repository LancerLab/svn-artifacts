module @f_3_cnn_256x512_512x10_256x10 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_3_cnn_256x512_512x10_256x10_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_3_cnn_256x512_512x10_256x10_dispatch_0_matmul_256x10x512_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_3_cnn_256x512_512x10_256x10_dispatch_0_matmul_256x10x512_f32() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x512xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512x10xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<256x10xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x512xf32>> -> tensor<256x512xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 10], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x10xf32>> -> tensor<512x10xf32>
          %5 = tensor.empty() : tensor<256x10xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x10xf32>) -> tensor<256x10xf32>
          %7 = linalg.matmul ins(%3, %4 : tensor<256x512xf32>, tensor<512x10xf32>) outs(%6 : tensor<256x10xf32>) -> tensor<256x10xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [256, 10], strides = [1, 1] : tensor<256x10xf32> -> !flow.dispatch.tensor<writeonly:tensor<256x10xf32>>
          return
        }
      }
    }
  }
  func.func @f_3_cnn_256x512_512x10_256x10(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c524288 = arith.constant 524288 : index
    %c20480 = arith.constant 20480 : index
    %c10240 = arith.constant 10240 : index
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c256, %c512]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<256x512xf32> in !stream.resource<external>{%c524288}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c512, %c10]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<512x10xf32> in !stream.resource<external>{%c20480}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c10240}
    %3 = stream.cmd.execute with(%0 as %arg2: !stream.resource<external>{%c524288}, %1 as %arg3: !stream.resource<external>{%c20480}, %2 as %arg4: !stream.resource<external>{%c10240}) {
      stream.cmd.dispatch @f_3_cnn_256x512_512x10_256x10_dispatch_0::@cuda_nvptx_fb::@f_3_cnn_256x512_512x10_256x10_dispatch_0_matmul_256x10x512_f32 {
        ro %arg2[%c0 for %c524288] : !stream.resource<external>{%c524288},
        ro %arg3[%c0 for %c20480] : !stream.resource<external>{%c20480},
        wo %arg4[%c0 for %c10240] : !stream.resource<external>{%c10240}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]}
    } => !stream.timepoint
    %4 = stream.timepoint.await %3 => %2 : !stream.resource<external>{%c10240}
    %5 = stream.tensor.export %4 : tensor<256x10xf32> in !stream.resource<external>{%c10240} -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
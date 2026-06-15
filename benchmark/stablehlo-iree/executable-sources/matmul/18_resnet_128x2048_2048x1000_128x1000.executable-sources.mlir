module @f_18_resnet_128x2048_2048x1000_128x1000 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_18_resnet_128x2048_2048x1000_128x1000_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_18_resnet_128x2048_2048x1000_128x1000_dispatch_0_matmul_128x1000x2048_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_18_resnet_128x2048_2048x1000_128x1000_dispatch_0_matmul_128x1000x2048_f32() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x2048xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1000xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x1000xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x2048xf32>> -> tensor<128x2048xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2048, 1000], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1000xf32>> -> tensor<2048x1000xf32>
          %5 = tensor.empty() : tensor<128x1000xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x1000xf32>) -> tensor<128x1000xf32>
          %7 = linalg.matmul ins(%3, %4 : tensor<128x2048xf32>, tensor<2048x1000xf32>) outs(%6 : tensor<128x1000xf32>) -> tensor<128x1000xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 1000], strides = [1, 1] : tensor<128x1000xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x1000xf32>>
          return
        }
      }
    }
  }
  func.func @f_18_resnet_128x2048_2048x1000_128x1000(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c1048576 = arith.constant 1048576 : index
    %c8192000 = arith.constant 8192000 : index
    %c512000 = arith.constant 512000 : index
    %c0 = arith.constant 0 : index
    %c1000 = arith.constant 1000 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128 = arith.constant 128 : index
    %c2048 = arith.constant 2048 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c128, %c2048]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<128x2048xf32> in !stream.resource<external>{%c1048576}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c2048, %c1000]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<2048x1000xf32> in !stream.resource<external>{%c8192000}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c512000}
    %3 = stream.cmd.execute with(%0 as %arg2: !stream.resource<external>{%c1048576}, %1 as %arg3: !stream.resource<external>{%c8192000}, %2 as %arg4: !stream.resource<external>{%c512000}) {
      stream.cmd.dispatch @f_18_resnet_128x2048_2048x1000_128x1000_dispatch_0::@cuda_nvptx_fb::@f_18_resnet_128x2048_2048x1000_128x1000_dispatch_0_matmul_128x1000x2048_f32 {
        ro %arg2[%c0 for %c1048576] : !stream.resource<external>{%c1048576},
        ro %arg3[%c0 for %c8192000] : !stream.resource<external>{%c8192000},
        wo %arg4[%c0 for %c512000] : !stream.resource<external>{%c512000}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]}
    } => !stream.timepoint
    %4 = stream.timepoint.await %3 => %2 : !stream.resource<external>{%c512000}
    %5 = stream.tensor.export %4 : tensor<128x1000xf32> in !stream.resource<external>{%c512000} -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
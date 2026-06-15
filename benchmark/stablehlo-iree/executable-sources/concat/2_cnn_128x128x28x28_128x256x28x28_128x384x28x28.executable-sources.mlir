module @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_dispatch_0 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_dispatch_0() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x128x28x28xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<128x384x28x28xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [128, 128, 28, 28], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x128x28x28xf32>> -> tensor<128x128x28x28xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 0, 0, 0], sizes = [128, 128, 28, 28], strides = [1, 1, 1, 1] : tensor<128x128x28x28xf32> -> !flow.dispatch.tensor<readwrite:tensor<128x384x28x28xf32>>
          return
        }
      }
    }
  }
  hal.executable private @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_dispatch_1 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_dispatch_1 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_dispatch_1() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x256x28x28xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<128x384x28x28xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [128, 256, 28, 28], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256x28x28xf32>> -> tensor<128x256x28x28xf32>
          flow.dispatch.tensor.store %2, %1, offsets = [0, 128, 0, 0], sizes = [128, 256, 28, 28], strides = [1, 1, 1, 1] : tensor<128x256x28x28xf32> -> !flow.dispatch.tensor<readwrite:tensor<128x384x28x28xf32>>
          return
        }
      }
    }
  }
  func.func @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c51380224 = arith.constant 51380224 : index
    %c102760448 = arith.constant 102760448 : index
    %c154140672 = arith.constant 154140672 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128 = arith.constant 128 : index
    %c28 = arith.constant 28 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c128, %c128, %c28, %c28]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<128x128x28x28xf32> in !stream.resource<external>{%c51380224}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c128, %c256, %c28, %c28]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<128x256x28x28xf32> in !stream.resource<external>{%c102760448}
    %2 = stream.resource.alloc uninitialized : !stream.resource<external>{%c154140672}
    %3 = stream.cmd.execute with(%0 as %arg2: !stream.resource<external>{%c51380224}, %1 as %arg3: !stream.resource<external>{%c102760448}, %2 as %arg4: !stream.resource<external>{%c154140672}) {
      stream.cmd.dispatch @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_dispatch_0::@cuda_nvptx_fb::@f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_dispatch_0 {
        ro %arg2[%c0 for %c51380224] : !stream.resource<external>{%c51380224},
        rw %arg4[%c0 for %c154140672] : !stream.resource<external>{%c154140672}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
      stream.cmd.dispatch @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_dispatch_1::@cuda_nvptx_fb::@f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_dispatch_1 {
        ro %arg3[%c0 for %c102760448] : !stream.resource<external>{%c102760448},
        rw %arg4[%c0 for %c154140672] : !stream.resource<external>{%c154140672}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %4 = stream.timepoint.await %3 => %2 : !stream.resource<external>{%c154140672}
    %5 = stream.tensor.export %4 : tensor<128x384x28x28xf32> in !stream.resource<external>{%c154140672} -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
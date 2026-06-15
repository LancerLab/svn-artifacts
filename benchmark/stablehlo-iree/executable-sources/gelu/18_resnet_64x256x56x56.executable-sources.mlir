module @f_18_resnet_64x256x56x56 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_18_resnet_64x256x56x56_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_18_resnet_64x256x56x56_dispatch_0_generic_51380224_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_18_resnet_64x256x56x56_dispatch_0_generic_51380224_f32() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 4.471500e-02 : f32
          %cst_0 = arith.constant 0.797884583 : f32
          %cst_1 = arith.constant 1.000000e+00 : f32
          %cst_2 = arith.constant 5.000000e-01 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<51380224xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<51380224xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [51380224], strides = [1] : !flow.dispatch.tensor<readonly:tensor<51380224xf32>> -> tensor<51380224xf32>
          %3 = tensor.empty() : tensor<51380224xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2 : tensor<51380224xf32>) outs(%3 : tensor<51380224xf32>) {
          ^bb0(%in: f32, %out: f32):
            %5 = arith.mulf %in, %in : f32
            %6 = arith.mulf %5, %in : f32
            %7 = arith.mulf %6, %cst : f32
            %8 = arith.addf %in, %7 : f32
            %9 = arith.mulf %8, %cst_0 : f32
            %10 = math.tanh %9 : f32
            %11 = arith.addf %10, %cst_1 : f32
            %12 = arith.mulf %in, %cst_2 : f32
            %13 = arith.mulf %12, %11 : f32
            linalg.yield %13 : f32
          } -> tensor<51380224xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0], sizes = [51380224], strides = [1] : tensor<51380224xf32> -> !flow.dispatch.tensor<writeonly:tensor<51380224xf32>>
          return
        }
      }
    }
  }
  func.func @f_18_resnet_64x256x56x56(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c205520896 = arith.constant 205520896 : index
    %c0 = arith.constant 0 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c56 = arith.constant 56 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c64, %c256, %c56, %c56]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<64x256x56x56xf32> in !stream.resource<external>{%c205520896}
    %1 = stream.resource.alloc uninitialized : !stream.resource<external>{%c205520896}
    %2 = stream.cmd.execute with(%0 as %arg1: !stream.resource<external>{%c205520896}, %1 as %arg2: !stream.resource<external>{%c205520896}) {
      stream.cmd.dispatch @f_18_resnet_64x256x56x56_dispatch_0::@cuda_nvptx_fb::@f_18_resnet_64x256x56x56_dispatch_0_generic_51380224_f32 {
        ro %arg1[%c0 for %c205520896] : !stream.resource<external>{%c205520896},
        wo %arg2[%c0 for %c205520896] : !stream.resource<external>{%c205520896}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %3 = stream.timepoint.await %2 => %1 : !stream.resource<external>{%c205520896}
    %4 = stream.tensor.export %3 : tensor<64x256x56x56xf32> in !stream.resource<external>{%c205520896} -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
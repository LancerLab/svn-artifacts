module @f_21_vit_32x3x224x224_32x224x224x3 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_21_vit_32x3x224x224_32x224x224x3_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_21_vit_32x3x224x224_32x224x224x3_dispatch_0_generic_32x50176x3_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_21_vit_32x3x224x224_32x224x224x3_dispatch_0_generic_32x50176x3_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x3x50176xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x50176x3xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [32, 3, 50176], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x3x50176xf32>> -> tensor<32x3x50176xf32>
          %3 = tensor.empty() : tensor<32x50176x3xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<32x3x50176xf32>) outs(%3 : tensor<32x50176x3xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<32x50176x3xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [32, 50176, 3], strides = [1, 1, 1] : tensor<32x50176x3xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x50176x3xf32>>
          return
        }
      }
    }
  }
  func.func @f_21_vit_32x3x224x224_32x224x224x3(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c19267584 = arith.constant 19267584 : index
    %c0 = arith.constant 0 : index
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32 = arith.constant 32 : index
    %c3 = arith.constant 3 : index
    %c224 = arith.constant 224 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c32, %c3, %c224, %c224]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<32x3x224x224xf32> in !stream.resource<external>{%c19267584}
    %1 = stream.resource.alloc uninitialized : !stream.resource<external>{%c19267584}
    %2 = stream.cmd.execute with(%0 as %arg1: !stream.resource<external>{%c19267584}, %1 as %arg2: !stream.resource<external>{%c19267584}) {
      stream.cmd.dispatch @f_21_vit_32x3x224x224_32x224x224x3_dispatch_0::@cuda_nvptx_fb::@f_21_vit_32x3x224x224_32x224x224x3_dispatch_0_generic_32x50176x3_f32 {
        ro %arg1[%c0 for %c19267584] : !stream.resource<external>{%c19267584},
        wo %arg2[%c0 for %c19267584] : !stream.resource<external>{%c19267584}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>]}
    } => !stream.timepoint
    %3 = stream.timepoint.await %2 => %1 : !stream.resource<external>{%c19267584}
    %4 = stream.tensor.export %3 : tensor<32x224x224x3xf32> in !stream.resource<external>{%c19267584} -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
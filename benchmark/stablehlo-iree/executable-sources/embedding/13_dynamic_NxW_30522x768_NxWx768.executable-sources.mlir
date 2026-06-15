module @f_13_dynamic_NxW_30522x768_NxWx768 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_13_dynamic_NxW_30522x768_NxWx768_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_13_dynamic_NxW_30522x768_NxWx768_dispatch_0_generic_DxDx768_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_13_dynamic_NxW_30522x768_NxWx768_dispatch_0_generic_DxDx768_f32() {
          %c30521 = arith.constant 30521 : index
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = arith.index_castui %0 : i32 to index
          %3 = arith.index_castui %1 : i32 to index
          %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<30522x768xf32>>
          %5 = flow.dispatch.workload.ordinal %2, 0 : index
          %6 = flow.dispatch.workload.ordinal %3, 1 : index
          %7 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%5, %6}
          %8 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x?x768xf32>>{%5, %6}
          %9 = flow.dispatch.tensor.load %7, offsets = [0, 0], sizes = [%5, %6], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%5, %6} -> tensor<?x?xi32>
          %10 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [30522, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<30522x768xf32>> -> tensor<30522x768xf32>
          %11 = tensor.empty(%5, %6) : tensor<?x?x768xf32>
          %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} outs(%11 : tensor<?x?x768xf32>) {
          ^bb0(%out: f32):
            %13 = linalg.index 0 : index
            %14 = linalg.index 1 : index
            %15 = linalg.index 2 : index
            %extracted = tensor.extract %9[%13, %14] : tensor<?x?xi32>
            %16 = arith.index_cast %extracted : i32 to index
            %17 = arith.maxsi %16, %c0 : index
            %18 = arith.minsi %17, %c30521 : index
            %extracted_0 = tensor.extract %10[%18, %15] : tensor<30522x768xf32>
            linalg.yield %extracted_0 : f32
          } -> tensor<?x?x768xf32>
          flow.dispatch.tensor.store %12, %8, offsets = [0, 0, 0], sizes = [%5, %6, 768], strides = [1, 1, 1] : tensor<?x?x768xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x768xf32>>{%5, %6}
          return
        }
      }
    }
  }
  func.func @f_13_dynamic_NxW_30522x768_NxWx768(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c93763584 = arith.constant 93763584 : index
    %c4 = arith.constant 4 : index
    %c3072 = arith.constant 3072 : index
    %c0 = arith.constant 0 : index
    %c268435488_i32 = arith.constant 268435488 : i32
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c30522 = arith.constant 30522 : index
    %c768 = arith.constant 768 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c30522, %c768]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<30522x768xf32> in !stream.resource<external>{%c93763584}
    %1 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
    %2 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%1, %2]) type(%c268435488_i32) encoding(%c1_i32)
    %3 = arith.muli %1, %c4 : index
    %4 = arith.muli %3, %2 : index
    %5 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<?x?xi32>{%1, %2} in !stream.resource<external>{%4}
    %6 = arith.muli %1, %c3072 : index
    %7 = arith.muli %6, %2 : index
    %8 = stream.resource.alloc uninitialized : !stream.resource<external>{%7}
    %9 = arith.index_castui %1 : index to i32
    %10 = arith.index_castui %2 : index to i32
    %11 = stream.cmd.execute with(%5 as %arg2: !stream.resource<external>{%4}, %0 as %arg3: !stream.resource<external>{%c93763584}, %8 as %arg4: !stream.resource<external>{%7}) {
      stream.cmd.dispatch @f_13_dynamic_NxW_30522x768_NxWx768_dispatch_0::@cuda_nvptx_fb::@f_13_dynamic_NxW_30522x768_NxWx768_dispatch_0_generic_DxDx768_f32[%1, %2](%9, %10 : i32, i32) {
        ro %arg2[%c0 for %4] : !stream.resource<external>{%4},
        ro %arg3[%c0 for %c93763584] : !stream.resource<external>{%c93763584},
        wo %arg4[%c0 for %7] : !stream.resource<external>{%7}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]}
    } => !stream.timepoint
    %12 = stream.timepoint.await %11 => %8 : !stream.resource<external>{%7}
    %13 = stream.tensor.export %12 : tensor<?x?x768xf32>{%1, %2} in !stream.resource<external>{%7} -> !hal.buffer_view
    return %13 : !hal.buffer_view
  }
}
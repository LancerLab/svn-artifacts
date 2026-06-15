module @f_11_dynamic_64xN_40000x512_64xNx512 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_11_dynamic_64xN_40000x512_64xNx512_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_11_dynamic_64xN_40000x512_64xNx512_dispatch_0_generic_64xDx512_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 1, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_11_dynamic_64xN_40000x512_64xNx512_dispatch_0_generic_64xDx512_f32() {
          %c39999 = arith.constant 39999 : index
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = arith.index_castui %0 : i32 to index
          %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<40000x512xf32>>
          %3 = flow.dispatch.workload.ordinal %1, 0 : index
          %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x?xi32>>{%3}
          %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x?x512xf32>>{%3}
          %6 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [64, %3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x?xi32>>{%3} -> tensor<64x?xi32>
          %7 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [40000, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<40000x512xf32>> -> tensor<40000x512xf32>
          %8 = tensor.empty(%3) : tensor<64x?x512xf32>
          %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} outs(%8 : tensor<64x?x512xf32>) {
          ^bb0(%out: f32):
            %10 = linalg.index 0 : index
            %11 = linalg.index 1 : index
            %12 = linalg.index 2 : index
            %extracted = tensor.extract %6[%10, %11] : tensor<64x?xi32>
            %13 = arith.index_cast %extracted : i32 to index
            %14 = arith.maxsi %13, %c0 : index
            %15 = arith.minsi %14, %c39999 : index
            %extracted_0 = tensor.extract %7[%15, %12] : tensor<40000x512xf32>
            linalg.yield %extracted_0 : f32
          } -> tensor<64x?x512xf32>
          flow.dispatch.tensor.store %9, %5, offsets = [0, 0, 0], sizes = [64, %3, 512], strides = [1, 1, 1] : tensor<64x?x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x?x512xf32>>{%3}
          return
        }
      }
    }
  }
  func.func @f_11_dynamic_64xN_40000x512_64xNx512(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c81920000 = arith.constant 81920000 : index
    %c256 = arith.constant 256 : index
    %c131072 = arith.constant 131072 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c268435488_i32 = arith.constant 268435488 : i32
    %c553648160_i32 = arith.constant 553648160 : i32
    %c1_i32 = arith.constant 1 : i32
    %c40000 = arith.constant 40000 : index
    %c512 = arith.constant 512 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c40000, %c512]) type(%c553648160_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<40000x512xf32> in !stream.resource<external>{%c81920000}
    %1 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c64, %1]) type(%c268435488_i32) encoding(%c1_i32)
    %2 = arith.muli %1, %c256 : index
    %3 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<64x?xi32>{%1} in !stream.resource<external>{%2}
    %4 = arith.muli %1, %c131072 : index
    %5 = stream.resource.alloc uninitialized : !stream.resource<external>{%4}
    %6 = arith.index_castui %1 : index to i32
    %7 = stream.cmd.execute with(%3 as %arg2: !stream.resource<external>{%2}, %0 as %arg3: !stream.resource<external>{%c81920000}, %5 as %arg4: !stream.resource<external>{%4}) {
      stream.cmd.dispatch @f_11_dynamic_64xN_40000x512_64xNx512_dispatch_0::@cuda_nvptx_fb::@f_11_dynamic_64xN_40000x512_64xNx512_dispatch_0_generic_64xDx512_f32[%1](%6 : i32) {
        ro %arg2[%c0 for %2] : !stream.resource<external>{%2},
        ro %arg3[%c0 for %c81920000] : !stream.resource<external>{%c81920000},
        wo %arg4[%c0 for %4] : !stream.resource<external>{%4}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]}
    } => !stream.timepoint
    %8 = stream.timepoint.await %7 => %5 : !stream.resource<external>{%4}
    %9 = stream.tensor.export %8 : tensor<64x?x512xf32>{%1} in !stream.resource<external>{%4} -> !hal.buffer_view
    return %9 : !hal.buffer_view
  }
}
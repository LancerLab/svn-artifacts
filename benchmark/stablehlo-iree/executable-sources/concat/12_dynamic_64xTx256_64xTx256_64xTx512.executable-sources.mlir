module @f_12_dynamic_64xTx256_64xTx256_64xTx512 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_12_dynamic_64xTx256_64xTx256_64xTx512_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_12_dynamic_64xTx256_64xTx256_64xTx512_dispatch_0_generic_64xDx512_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_12_dynamic_64xTx256_64xTx256_64xTx512_dispatch_0_generic_64xDx512_f32() {
          %c256 = arith.constant 256 : index
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = arith.index_castui %0 : i32 to index
          %3 = arith.index_castui %1 : i32 to index
          %4 = flow.dispatch.workload.ordinal %2, 0 : index
          %5 = flow.dispatch.workload.ordinal %3, 1 : index
          %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x?x256xf32>>{%5}
          %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x?x256xf32>>{%4}
          %8 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x?x512xf32>>{%5}
          %9 = flow.dispatch.tensor.load %6, offsets = [0, 0, 0], sizes = [64, %5, 256], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x?x256xf32>>{%5} -> tensor<64x?x256xf32>
          %10 = flow.dispatch.tensor.load %7, offsets = [0, 0, 0], sizes = [64, %4, 256], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x?x256xf32>>{%4} -> tensor<64x?x256xf32>
          %11 = tensor.empty(%5) : tensor<64x?x512xf32>
          %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} outs(%11 : tensor<64x?x512xf32>) {
          ^bb0(%out: f32):
            %13 = linalg.index 0 : index
            %14 = linalg.index 1 : index
            %15 = linalg.index 2 : index
            %16 = arith.cmpi ult, %15, %c256 : index
            %17 = scf.if %16 -> (f32) {
              %extracted = tensor.extract %9[%13, %14, %15] : tensor<64x?x256xf32>
              scf.yield %extracted : f32
            } else {
              %18 = arith.subi %15, %c256 : index
              %extracted = tensor.extract %10[%13, %14, %18] : tensor<64x?x256xf32>
              scf.yield %extracted : f32
            }
            linalg.yield %17 : f32
          } -> tensor<64x?x512xf32>
          flow.dispatch.tensor.store %12, %8, offsets = [0, 0, 0], sizes = [64, %5, 512], strides = [1, 1, 1] : tensor<64x?x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x?x512xf32>>{%5}
          return
        }
      }
    }
  }
  func.func @f_12_dynamic_64xTx256_64xTx256_64xTx512(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c65536 = arith.constant 65536 : index
    %c131072 = arith.constant 131072 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %c1_i32 = arith.constant 1 : i32
    %c553648160_i32 = arith.constant 553648160 : i32
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c64, %0, %c256]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = arith.muli %0, %c65536 : index
    %2 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<64x?x256xf32>{%0} in !stream.resource<external>{%1}
    %3 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c64, %3, %c256]) type(%c553648160_i32) encoding(%c1_i32)
    %4 = arith.muli %3, %c65536 : index
    %5 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<64x?x256xf32>{%3} in !stream.resource<external>{%4}
    %6 = arith.muli %0, %c131072 : index
    %7 = stream.resource.alloc uninitialized : !stream.resource<external>{%6}
    %8 = arith.index_castui %3 : index to i32
    %9 = arith.index_castui %0 : index to i32
    %10 = stream.cmd.execute with(%2 as %arg2: !stream.resource<external>{%1}, %5 as %arg3: !stream.resource<external>{%4}, %7 as %arg4: !stream.resource<external>{%6}) {
      stream.cmd.dispatch @f_12_dynamic_64xTx256_64xTx256_64xTx512_dispatch_0::@cuda_nvptx_fb::@f_12_dynamic_64xTx256_64xTx256_64xTx512_dispatch_0_generic_64xDx512_f32[%3, %0](%8, %9 : i32, i32) {
        ro %arg2[%c0 for %1] : !stream.resource<external>{%1},
        ro %arg3[%c0 for %4] : !stream.resource<external>{%4},
        wo %arg4[%c0 for %6] : !stream.resource<external>{%6}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]}
    } => !stream.timepoint
    %11 = stream.timepoint.await %10 => %7 : !stream.resource<external>{%6}
    %12 = stream.tensor.export %11 : tensor<64x?x512xf32>{%0} in !stream.resource<external>{%6} -> !hal.buffer_view
    return %12 : !hal.buffer_view
  }
}
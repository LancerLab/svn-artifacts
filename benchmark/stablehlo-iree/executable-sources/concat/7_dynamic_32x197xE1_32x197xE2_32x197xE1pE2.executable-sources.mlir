module @f_7_dynamic_32x197xE1_32x197xE2_32x197xE1pE2 attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>], legacy_sync}>]} {
  hal.executable private @f_7_dynamic_32x197xE1_32x197xE2_32x197xE1pE2_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}> {
      hal.executable.export public @f_7_dynamic_32x197xE1_32x197xE2_32x197xE1pE2_dispatch_0_generic_32x197xD_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 3, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_7_dynamic_32x197xE1_32x197xE2_32x197xE1pE2_dispatch_0_generic_32x197xD_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = hal.interface.constant.load[2] : i32
          %3 = arith.index_castui %0 : i32 to index
          %4 = arith.index_castui %1 : i32 to index
          %5 = arith.index_castui %2 : i32 to index
          %6 = flow.dispatch.workload.ordinal %3, 0 : index
          %7 = flow.dispatch.workload.ordinal %4, 1 : index
          %8 = flow.dispatch.workload.ordinal %5, 2 : index
          %9 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x197x?xf32>>{%6}
          %10 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x197x?xf32>>{%7}
          %11 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x197x?xf32>>{%8}
          %12 = flow.dispatch.tensor.load %9, offsets = [0, 0, 0], sizes = [32, 197, %6], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x197x?xf32>>{%6} -> tensor<32x197x?xf32>
          %13 = flow.dispatch.tensor.load %10, offsets = [0, 0, 0], sizes = [32, 197, %7], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x197x?xf32>>{%7} -> tensor<32x197x?xf32>
          %14 = tensor.empty(%8) : tensor<32x197x?xf32>
          %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} outs(%14 : tensor<32x197x?xf32>) {
          ^bb0(%out: f32):
            %16 = linalg.index 0 : index
            %17 = linalg.index 1 : index
            %18 = linalg.index 2 : index
            %19 = arith.cmpi ult, %18, %6 : index
            %20 = scf.if %19 -> (f32) {
              %extracted = tensor.extract %12[%16, %17, %18] : tensor<32x197x?xf32>
              scf.yield %extracted : f32
            } else {
              %21 = arith.subi %18, %6 : index
              %extracted = tensor.extract %13[%16, %17, %21] : tensor<32x197x?xf32>
              scf.yield %extracted : f32
            }
            linalg.yield %20 : f32
          } -> tensor<32x197x?xf32>
          flow.dispatch.tensor.store %15, %11, offsets = [0, 0, 0], sizes = [32, 197, %8], strides = [1, 1, 1] : tensor<32x197x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x197x?xf32>>{%8}
          return
        }
      }
    }
  }
  func.func @f_7_dynamic_32x197xE1_32x197xE2_32x197xE1pE2(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c25216 = arith.constant 25216 : index
    %c0 = arith.constant 0 : index
    %c197 = arith.constant 197 : index
    %c32 = arith.constant 32 : index
    %c1_i32 = arith.constant 1 : i32
    %c553648160_i32 = arith.constant 553648160 : i32
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[2] : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c32, %c197, %0]) type(%c553648160_i32) encoding(%c1_i32)
    %1 = arith.muli %0, %c25216 : index
    %2 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<32x197x?xf32>{%0} in !stream.resource<external>{%1}
    %3 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[2] : index
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input 1") shape([%c32, %c197, %3]) type(%c553648160_i32) encoding(%c1_i32)
    %4 = arith.muli %3, %c25216 : index
    %5 = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<32x197x?xf32>{%3} in !stream.resource<external>{%4}
    %6 = arith.addi %0, %3 : index
    %7 = arith.muli %6, %c25216 : index
    %8 = stream.resource.alloc uninitialized : !stream.resource<external>{%7}
    %9 = arith.index_castui %0 : index to i32
    %10 = arith.index_castui %3 : index to i32
    %11 = arith.index_castui %6 : index to i32
    %12 = stream.cmd.execute with(%2 as %arg2: !stream.resource<external>{%1}, %5 as %arg3: !stream.resource<external>{%4}, %8 as %arg4: !stream.resource<external>{%7}) {
      stream.cmd.dispatch @f_7_dynamic_32x197xE1_32x197xE2_32x197xE1pE2_dispatch_0::@cuda_nvptx_fb::@f_7_dynamic_32x197xE1_32x197xE2_32x197xE1pE2_dispatch_0_generic_32x197xD_f32[%0, %3, %6](%9, %10, %11 : i32, i32, i32) {
        ro %arg2[%c0 for %1] : !stream.resource<external>{%1},
        ro %arg3[%c0 for %4] : !stream.resource<external>{%4},
        wo %arg4[%c0 for %7] : !stream.resource<external>{%7}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]}
    } => !stream.timepoint
    %13 = stream.timepoint.await %12 => %8 : !stream.resource<external>{%7}
    %14 = stream.tensor.export %13 : tensor<32x197x?xf32>{%6} in !stream.resource<external>{%7} -> !hal.buffer_view
    return %14 : !hal.buffer_view
  }
}
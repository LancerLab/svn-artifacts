module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #hal.device.target<"cuda", [#hal.executable.target<"cuda", "cuda-nvptx-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "sm_60", features = "+ptx76", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 49152, max_workgroup_counts = [2147483647, 65535, 65535]>>}>]> : !hal.device
  hal.executable private @f_6_dynamic_128xCx112x112_128x112x112xC_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "sm_60", features = "+ptx76", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 49152, max_workgroup_counts = [2147483647, 65535, 65535]>>}>) {
      hal.executable.export public @f_6_dynamic_128xCx112x112_128x112x112xC_dispatch_0_transpose_128xDx12544_f32 ordinal(0) layout(#hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device, %arg1: index) -> (index, index, index) {
        %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg1)
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @f_6_dynamic_128xCx112x112_128x112x112xC_dispatch_0_transpose_128xDx12544_f32() {
          %c32_i64 = arith.constant 32 : i64
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
          %1 = hal.interface.constant.load layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
          %2 = arith.extui %1 : i32 to i64
          %3 = arith.shli %2, %c32_i64 : i64
          %4 = arith.extui %0 : i32 to i64
          %5 = arith.ori %4, %3 : i64
          %6 = arith.index_castui %5 : i64 to index
          %7 = util.assume.int %6<umin = 0, umax = 9007199254740991> : index
          %8 = iree_tensor_ext.dispatch.workload.ordinal %7, 0 : index
          %9 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x?x12544xf32>>{%8}
          %10 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x12544x?xf32>>{%8}
          %11 = iree_tensor_ext.dispatch.tensor.load %9, offsets = [0, 0, 0], sizes = [128, %8, 12544], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x?x12544xf32>>{%8} -> tensor<128x?x12544xf32>
          %12 = tensor.empty(%8) : tensor<128x12544x?xf32>
          %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2, d1)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11 : tensor<128x?x12544xf32>) outs(%12 : tensor<128x12544x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<128x12544x?xf32>
          iree_tensor_ext.dispatch.tensor.store %13, %10, offsets = [0, 0, 0], sizes = [128, 12544, %8], strides = [1, 1, 1] : tensor<128x12544x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x12544x?xf32>>{%8}
          return
        }
      }
    }
  }
  util.func public @f_6_dynamic_128xCx112x112_128x112x112xC(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @f_6_dynamic_128xCx112x112_128x112x112xC(%input0: tensor<128x?x112x112xf32>) -> (%output0: tensor<128x112x112x?xf32>)"}} {
    %c32_i64 = arith.constant 32 : i64
    %c0 = arith.constant 0 : index
    %c6422528 = arith.constant 6422528 : index
    %c112 = arith.constant 112 : index
    %c128 = arith.constant 128 : index
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
    %element_type_f32 = hal.element_type<f32> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%c128, %0, %c112, %c112]) type(%element_type_f32) encoding(%dense_row_major)
    %1 = arith.muli %0, %c6422528 : index
    %2 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg0 : !hal.buffer_view -> tensor<128x?x112x112xf32>{%0} in !stream.resource<external>{%1}
    %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<external>{%1} => !stream.timepoint
    %3 = arith.index_castui %0 : index to i64
    %4 = arith.index_castui %0 : index to i32
    %5 = arith.shrui %3, %c32_i64 : i64
    %6 = arith.trunci %5 : i64 to i32
    %7 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%result_timepoint) => with(%2 as %arg1: !stream.resource<external>{%1}, %result as %arg2: !stream.resource<external>{%1}) {
      stream.cmd.dispatch @f_6_dynamic_128xCx112x112_128x112x112xC_dispatch_0::@cuda_nvptx_fb::@f_6_dynamic_128xCx112x112_128x112x112xC_dispatch_0_transpose_128xDx12544_f32[%0](%4, %6 : i32, i32) {
        ro %arg1[%c0 for %1] : !stream.resource<external>{%1},
        wo %arg2[%c0 for %1] : !stream.resource<external>{%1}
      }
    } => !stream.timepoint
    %8 = stream.timepoint.await %7 => %result : !stream.resource<external>{%1}
    %9 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %8 : tensor<128x112x112x?xf32>{%0} in !stream.resource<external>{%1} -> !hal.buffer_view
    util.return %9 : !hal.buffer_view
  }
}
module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #hal.device.target<"cuda", [#hal.executable.target<"cuda", "cuda-nvptx-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "sm_60", features = "+ptx76", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 49152, max_workgroup_counts = [2147483647, 65535, 65535]>>}>]> : !hal.device
  util.func public @f_8_dynamic_16x1024xD_16x1024xDd64x64(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @f_8_dynamic_16x1024xD_16x1024xDd64x64(%input0: tensor<16x1024x?xf32>) -> (%output0: tensor<16x1024x?x64xf32>)"}} {
    %c4194304 = arith.constant 4194304 : index
    %c65536 = arith.constant 65536 : index
    %c1024 = arith.constant 1024 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[2] : index
    %element_type_f32 = hal.element_type<f32> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%c16, %c1024, %0]) type(%element_type_f32) encoding(%dense_row_major)
    %1 = arith.muli %0, %c65536 : index
    %2 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg0 : !hal.buffer_view -> tensor<16x1024x?xf32>{%0} in !stream.resource<external>{%1}
    %3 = arith.divui %0, %c64 : index
    %4 = arith.muli %3, %c4194304 : index
    %5 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %2 : tensor<16x1024x?x64xf32>{%3} in !stream.resource<external>{%4} -> !hal.buffer_view
    util.return %5 : !hal.buffer_view
  }
}
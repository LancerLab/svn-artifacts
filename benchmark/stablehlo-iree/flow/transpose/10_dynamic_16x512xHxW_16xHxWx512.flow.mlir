module @f_10_dynamic_16x512xHxW_16xHxWx512 {
  flow.executable private @f_10_dynamic_16x512xHxW_16xHxWx512_dispatch_0 {
    flow.executable.export public @f_10_dynamic_16x512xHxW_16xHxWx512_dispatch_0_generic_16xDxDx512_f32 workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_10_dynamic_16x512xHxW_16xHxWx512_dispatch_0_generic_16xDxDx512_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<16x512x?x?xf32>>, %arg1: index, %arg2: index, %arg3: !flow.dispatch.tensor<writeonly:tensor<16x?x?x512xf32>>) {
        %0 = flow.dispatch.workload.ordinal %arg1, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg2, 1 : index
        %2 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<16x512x?x?xf32>>{%0, %1}
        %3 = flow.dispatch.tie_shape %arg3 : !flow.dispatch.tensor<writeonly:tensor<16x?x?x512xf32>>{%0, %1}
        %4 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [16, 512, %0, %1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x512x?x?xf32>>{%0, %1} -> tensor<16x512x?x?xf32>
        %5 = tensor.empty(%0, %1) : tensor<16x?x?x512xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<16x512x?x?xf32>) outs(%5 : tensor<16x?x?x512xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<16x?x?x512xf32>
        flow.dispatch.tensor.store %6, %3, offsets = [0, 0, 0, 0], sizes = [16, %0, %1, 512], strides = [1, 1, 1, 1] : tensor<16x?x?x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x?x?x512xf32>>{%0, %1}
        return
      }
    }
  }
  func.func @f_10_dynamic_16x512xHxW_16xHxWx512(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[2] : index
    %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[3] : index
    %2 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<16x512x?x?xf32>{%0, %1}
    %3 = flow.dispatch @f_10_dynamic_16x512xHxW_16xHxWx512_dispatch_0::@f_10_dynamic_16x512xHxW_16xHxWx512_dispatch_0_generic_16xDxDx512_f32[%0, %1](%2, %0, %1) : (tensor<16x512x?x?xf32>{%0, %1}, index, index) -> tensor<16x?x?x512xf32>{%0, %1}
    %4 = hal.tensor.export %3 "output 0" : tensor<16x?x?x512xf32>{%0, %1} -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
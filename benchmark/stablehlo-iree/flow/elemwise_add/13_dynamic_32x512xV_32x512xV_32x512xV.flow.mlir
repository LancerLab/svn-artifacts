module @f_13_dynamic_32x512xV_32x512xV_32x512xV {
  flow.executable private @f_13_dynamic_32x512xV_32x512xV_32x512xV_dispatch_0 {
    flow.executable.export public @f_13_dynamic_32x512xV_32x512xV_32x512xV_dispatch_0_generic_32x512xD_f32 workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_13_dynamic_32x512xV_32x512xV_32x512xV_dispatch_0_generic_32x512xD_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<32x512x?xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<32x512x?xf32>>, %arg2: index, %arg3: index, %arg4: !flow.dispatch.tensor<writeonly:tensor<32x512x?xf32>>) {
        %0 = flow.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg3, 1 : index
        %2 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<32x512x?xf32>>{%1}
        %3 = flow.dispatch.tie_shape %arg1 : !flow.dispatch.tensor<readonly:tensor<32x512x?xf32>>{%0}
        %4 = flow.dispatch.tie_shape %arg4 : !flow.dispatch.tensor<writeonly:tensor<32x512x?xf32>>{%1}
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [32, 512, %1], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x512x?xf32>>{%1} -> tensor<32x512x?xf32>
        %6 = flow.dispatch.tensor.load %3, offsets = [0, 0, 0], sizes = [32, 512, %0], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x512x?xf32>>{%0} -> tensor<32x512x?xf32>
        %7 = tensor.empty(%1) : tensor<32x512x?xf32>
        %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %6 : tensor<32x512x?xf32>, tensor<32x512x?xf32>) outs(%7 : tensor<32x512x?xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %9 = arith.addf %in, %in_0 : f32
          linalg.yield %9 : f32
        } -> tensor<32x512x?xf32>
        flow.dispatch.tensor.store %8, %4, offsets = [0, 0, 0], sizes = [32, 512, %1], strides = [1, 1, 1] : tensor<32x512x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x512x?xf32>>{%1}
        return
      }
    }
  }
  func.func @f_13_dynamic_32x512xV_32x512xV_32x512xV(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[2] : index
    %1 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<32x512x?xf32>{%0}
    %2 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[2] : index
    %3 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<32x512x?xf32>{%2}
    %4 = flow.dispatch @f_13_dynamic_32x512xV_32x512xV_32x512xV_dispatch_0::@f_13_dynamic_32x512xV_32x512xV_32x512xV_dispatch_0_generic_32x512xD_f32[%2, %0](%1, %3, %2, %0) : (tensor<32x512x?xf32>{%0}, tensor<32x512x?xf32>{%2}, index, index) -> tensor<32x512x?xf32>{%0}
    %5 = hal.tensor.export %4 "output 0" : tensor<32x512x?xf32>{%0} -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
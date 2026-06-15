module @f_10_dynamic_128x1280_1280xN_128xN {
  flow.executable private @f_10_dynamic_128x1280_1280xN_128xN_dispatch_0 {
    flow.executable.export public @f_10_dynamic_128x1280_1280xN_128xN_dispatch_0_matmul_128xDx1280_f32 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_10_dynamic_128x1280_1280xN_128xN_dispatch_0_matmul_128xDx1280_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<128x1280xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<1280x?xf32>>, %arg2: index, %arg3: !flow.dispatch.tensor<writeonly:tensor<128x?xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = flow.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.tie_shape %arg1 : !flow.dispatch.tensor<readonly:tensor<1280x?xf32>>{%0}
        %2 = flow.dispatch.tie_shape %arg3 : !flow.dispatch.tensor<writeonly:tensor<128x?xf32>>{%0}
        %3 = flow.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [128, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x1280xf32>> -> tensor<128x1280xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1280, %0], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1280x?xf32>>{%0} -> tensor<1280x?xf32>
        %5 = tensor.empty(%0) : tensor<128x?xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x?xf32>) -> tensor<128x?xf32>
        %7 = linalg.matmul ins(%3, %4 : tensor<128x1280xf32>, tensor<1280x?xf32>) outs(%6 : tensor<128x?xf32>) -> tensor<128x?xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, %0], strides = [1, 1] : tensor<128x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x?xf32>>{%0}
        return
      }
    }
  }
  func.func @f_10_dynamic_128x1280_1280xN_128xN(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<128x1280xf32>
    %1 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
    %2 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<1280x?xf32>{%1}
    %3 = flow.dispatch @f_10_dynamic_128x1280_1280xN_128xN_dispatch_0::@f_10_dynamic_128x1280_1280xN_128xN_dispatch_0_matmul_128xDx1280_f32[%1](%0, %2, %1) : (tensor<128x1280xf32>, tensor<1280x?xf32>{%1}, index) -> tensor<128x?xf32>{%1}
    %4 = hal.tensor.export %3 "output 0" : tensor<128x?xf32>{%1} -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
module @f_8_dynamic_16x1024xD_16x1024xD {
  flow.executable private @f_8_dynamic_16x1024xD_16x1024xD_dispatch_0 {
    flow.executable.export public @f_8_dynamic_16x1024xD_16x1024xD_dispatch_0_generic_16x1024xD_f32 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_8_dynamic_16x1024xD_16x1024xD_dispatch_0_generic_16x1024xD_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<16x1024x?xf32>>, %arg1: index, %arg2: !flow.dispatch.tensor<writeonly:tensor<16x1024x?xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 3.40282347E+38 : f32
        %0 = flow.dispatch.workload.ordinal %arg1, 0 : index
        %1 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<16x1024x?xf32>>{%0}
        %2 = flow.dispatch.tie_shape %arg2 : !flow.dispatch.tensor<writeonly:tensor<16x1024x?xf32>>{%0}
        %3 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [16, 1024, %0], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x1024x?xf32>>{%0} -> tensor<16x1024x?xf32>
        %4 = tensor.empty(%0) : tensor<16x1024x?xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3 : tensor<16x1024x?xf32>) outs(%4 : tensor<16x1024x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %6 = arith.maxf %in, %cst : f32
          %7 = arith.minf %6, %cst_0 : f32
          linalg.yield %7 : f32
        } -> tensor<16x1024x?xf32>
        flow.dispatch.tensor.store %5, %2, offsets = [0, 0, 0], sizes = [16, 1024, %0], strides = [1, 1, 1] : tensor<16x1024x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x1024x?xf32>>{%0}
        return
      }
    }
  }
  func.func @f_8_dynamic_16x1024xD_16x1024xD(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[2] : index
    %1 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<16x1024x?xf32>{%0}
    %2 = flow.dispatch @f_8_dynamic_16x1024xD_16x1024xD_dispatch_0::@f_8_dynamic_16x1024xD_16x1024xD_dispatch_0_generic_16x1024xD_f32[%0](%1, %0) : (tensor<16x1024x?xf32>{%0}, index) -> tensor<16x1024x?xf32>{%0}
    %3 = hal.tensor.export %2 "output 0" : tensor<16x1024x?xf32>{%0} -> !hal.buffer_view
    return %3 : !hal.buffer_view
  }
}
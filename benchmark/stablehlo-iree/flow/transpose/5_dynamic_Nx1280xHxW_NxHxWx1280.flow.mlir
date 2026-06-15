module @f_5_dynamic_Nx1280xHxW_NxHxWx1280 {
  flow.executable private @f_5_dynamic_Nx1280xHxW_NxHxWx1280_dispatch_0 {
    flow.executable.export public @f_5_dynamic_Nx1280xHxW_NxHxWx1280_dispatch_0_generic_DxDxDx1280_f32 workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1, %arg2
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_5_dynamic_Nx1280xHxW_NxHxWx1280_dispatch_0_generic_DxDxDx1280_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<?x1280x?x?xf32>>, %arg1: index, %arg2: index, %arg3: index, %arg4: !flow.dispatch.tensor<writeonly:tensor<?x?x?x1280xf32>>) {
        %0 = flow.dispatch.workload.ordinal %arg1, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg2, 1 : index
        %2 = flow.dispatch.workload.ordinal %arg3, 2 : index
        %3 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x1280x?x?xf32>>{%0, %1, %2}
        %4 = flow.dispatch.tie_shape %arg4 : !flow.dispatch.tensor<writeonly:tensor<?x?x?x1280xf32>>{%0, %1, %2}
        %5 = flow.dispatch.tensor.load %3, offsets = [0, 0, 0, 0], sizes = [%0, 1280, %1, %2], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x1280x?x?xf32>>{%0, %1, %2} -> tensor<?x1280x?x?xf32>
        %6 = tensor.empty(%0, %1, %2) : tensor<?x?x?x1280xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5 : tensor<?x1280x?x?xf32>) outs(%6 : tensor<?x?x?x1280xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<?x?x?x1280xf32>
        flow.dispatch.tensor.store %7, %4, offsets = [0, 0, 0, 0], sizes = [%0, %1, %2, 1280], strides = [1, 1, 1, 1] : tensor<?x?x?x1280xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x?x1280xf32>>{%0, %1, %2}
        return
      }
    }
  }
  func.func @f_5_dynamic_Nx1280xHxW_NxHxWx1280(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[2] : index
    %2 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[3] : index
    %3 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<?x1280x?x?xf32>{%0, %1, %2}
    %4 = flow.dispatch @f_5_dynamic_Nx1280xHxW_NxHxWx1280_dispatch_0::@f_5_dynamic_Nx1280xHxW_NxHxWx1280_dispatch_0_generic_DxDxDx1280_f32[%0, %1, %2](%3, %0, %1, %2) : (tensor<?x1280x?x?xf32>{%0, %1, %2}, index, index, index) -> tensor<?x?x?x1280xf32>{%0, %1, %2}
    %5 = hal.tensor.export %4 "output 0" : tensor<?x?x?x1280xf32>{%0, %1, %2} -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
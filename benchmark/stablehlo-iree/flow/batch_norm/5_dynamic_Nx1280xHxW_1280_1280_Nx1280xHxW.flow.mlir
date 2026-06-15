module @f_5_dynamic_Nx1280xHxW_1280_1280_Nx1280xHxW {
  util.global private @hoisted = dense<1.00000501> : tensor<1280xf32>
  flow.executable private @f_5_dynamic_Nx1280xHxW_1280_1280_Nx1280xHxW_dispatch_0 {
    flow.executable.export public @f_5_dynamic_Nx1280xHxW_1280_1280_Nx1280xHxW_dispatch_0_generic_Dx1280xDxD_f32 workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1, %arg2
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_5_dynamic_Nx1280xHxW_1280_1280_Nx1280xHxW_dispatch_0_generic_Dx1280xDxD_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<?x1280x?x?xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<1280xf32>>, %arg2: !flow.dispatch.tensor<readonly:tensor<1280xf32>>, %arg3: !flow.dispatch.tensor<readonly:tensor<1280xf32>>, %arg4: index, %arg5: index, %arg6: index, %arg7: !flow.dispatch.tensor<writeonly:tensor<?x1280x?x?xf32>>) {
        %0 = flow.dispatch.workload.ordinal %arg4, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg5, 1 : index
        %2 = flow.dispatch.workload.ordinal %arg6, 2 : index
        %3 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x1280x?x?xf32>>{%0, %1, %2}
        %4 = flow.dispatch.tie_shape %arg7 : !flow.dispatch.tensor<writeonly:tensor<?x1280x?x?xf32>>{%0, %1, %2}
        %5 = flow.dispatch.tensor.load %3, offsets = [0, 0, 0, 0], sizes = [%0, 1280, %1, %2], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x1280x?x?xf32>>{%0, %1, %2} -> tensor<?x1280x?x?xf32>
        %6 = flow.dispatch.tensor.load %arg1, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf32>> -> tensor<1280xf32>
        %7 = flow.dispatch.tensor.load %arg2, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf32>> -> tensor<1280xf32>
        %8 = flow.dispatch.tensor.load %arg3, offsets = [0], sizes = [1280], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1280xf32>> -> tensor<1280xf32>
        %9 = tensor.empty(%0, %1, %2) : tensor<?x1280x?x?xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5, %6, %7, %8 : tensor<?x1280x?x?xf32>, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>) outs(%9 : tensor<?x1280x?x?xf32>) {
        ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
          %11 = arith.mulf %in, %in_0 : f32
          %12 = arith.divf %11, %in_1 : f32
          %13 = arith.addf %12, %in_2 : f32
          linalg.yield %13 : f32
        } -> tensor<?x1280x?x?xf32>
        flow.dispatch.tensor.store %10, %4, offsets = [0, 0, 0, 0], sizes = [%0, 1280, %1, %2], strides = [1, 1, 1, 1] : tensor<?x1280x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x1280x?x?xf32>>{%0, %1, %2}
        return
      }
    }
  }
  func.func @f_5_dynamic_Nx1280xHxW_1280_1280_Nx1280xHxW(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[2] : index
    %2 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[3] : index
    %3 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<?x1280x?x?xf32>{%0, %1, %2}
    %4 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<1280xf32>
    %5 = hal.tensor.import %arg2 "input 2" : !hal.buffer_view -> tensor<1280xf32>
    %hoisted = util.global.load @hoisted : tensor<1280xf32>
    %6 = flow.dispatch @f_5_dynamic_Nx1280xHxW_1280_1280_Nx1280xHxW_dispatch_0::@f_5_dynamic_Nx1280xHxW_1280_1280_Nx1280xHxW_dispatch_0_generic_Dx1280xDxD_f32[%0, %1, %2](%3, %4, %hoisted, %5, %0, %1, %2) : (tensor<?x1280x?x?xf32>{%0, %1, %2}, tensor<1280xf32>, tensor<1280xf32>, tensor<1280xf32>, index, index, index) -> tensor<?x1280x?x?xf32>{%0, %1, %2}
    %7 = hal.tensor.export %6 "output 0" : tensor<?x1280x?x?xf32>{%0, %1, %2} -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
}
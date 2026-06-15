module @f_13_dynamic_NxW_30522x768_NxWx768 {
  flow.executable private @f_13_dynamic_NxW_30522x768_NxWx768_dispatch_0 {
    flow.executable.export public @f_13_dynamic_NxW_30522x768_NxWx768_dispatch_0_generic_DxDx768_f32 workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_13_dynamic_NxW_30522x768_NxWx768_dispatch_0_generic_DxDx768_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<?x?xi32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<30522x768xf32>>, %arg2: index, %arg3: index, %arg4: !flow.dispatch.tensor<writeonly:tensor<?x?x768xf32>>) {
        %c0 = arith.constant 0 : index
        %c30521 = arith.constant 30521 : index
        %0 = flow.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg3, 1 : index
        %2 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%0, %1}
        %3 = flow.dispatch.tie_shape %arg4 : !flow.dispatch.tensor<writeonly:tensor<?x?x768xf32>>{%0, %1}
        %4 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%0, %1} -> tensor<?x?xi32>
        %5 = flow.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [30522, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<30522x768xf32>> -> tensor<30522x768xf32>
        %6 = tensor.empty(%0, %1) : tensor<?x?x768xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} outs(%6 : tensor<?x?x768xf32>) {
        ^bb0(%out: f32):
          %8 = linalg.index 0 : index
          %9 = linalg.index 1 : index
          %10 = linalg.index 2 : index
          %extracted = tensor.extract %4[%8, %9] : tensor<?x?xi32>
          %11 = arith.index_cast %extracted : i32 to index
          %12 = arith.maxsi %11, %c0 : index
          %13 = arith.minsi %12, %c30521 : index
          %extracted_0 = tensor.extract %5[%13, %10] : tensor<30522x768xf32>
          linalg.yield %extracted_0 : f32
        } -> tensor<?x?x768xf32>
        flow.dispatch.tensor.store %7, %3, offsets = [0, 0, 0], sizes = [%0, %1, 768], strides = [1, 1, 1] : tensor<?x?x768xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x768xf32>>{%0, %1}
        return
      }
    }
  }
  func.func @f_13_dynamic_NxW_30522x768_NxWx768(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<30522x768xf32>
    %1 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
    %2 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
    %3 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<?x?xi32>{%1, %2}
    %4 = flow.dispatch @f_13_dynamic_NxW_30522x768_NxWx768_dispatch_0::@f_13_dynamic_NxW_30522x768_NxWx768_dispatch_0_generic_DxDx768_f32[%1, %2](%3, %0, %1, %2) : (tensor<?x?xi32>{%1, %2}, tensor<30522x768xf32>, index, index) -> tensor<?x?x768xf32>{%1, %2}
    %5 = hal.tensor.export %4 "output 0" : tensor<?x?x768xf32>{%1, %2} -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
module @f_4_dynamic_Nx512_Vx768_Nx512x768 {
  flow.executable private @f_4_dynamic_Nx512_Vx768_Nx512x768_dispatch_0 {
    flow.executable.export public @f_4_dynamic_Nx512_Vx768_Nx512x768_dispatch_0_generic_Dx512x768_f32 workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_4_dynamic_Nx512_Vx768_Nx512x768_dispatch_0_generic_Dx512x768_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<?x512xi32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<?x768xf32>>, %arg2: index, %arg3: index, %arg4: !flow.dispatch.tensor<writeonly:tensor<?x512x768xf32>>) {
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %0 = flow.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg3, 1 : index
        %2 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x512xi32>>{%1}
        %3 = flow.dispatch.tie_shape %arg1 : !flow.dispatch.tensor<readonly:tensor<?x768xf32>>{%0}
        %4 = flow.dispatch.tie_shape %arg4 : !flow.dispatch.tensor<writeonly:tensor<?x512x768xf32>>{%1}
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%1, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x512xi32>>{%1} -> tensor<?x512xi32>
        %6 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [%0, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x768xf32>>{%0} -> tensor<?x768xf32>
        %7 = tensor.empty(%1) : tensor<?x512x768xf32>
        %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} outs(%7 : tensor<?x512x768xf32>) {
        ^bb0(%out: f32):
          %9 = linalg.index 0 : index
          %10 = linalg.index 1 : index
          %11 = linalg.index 2 : index
          %extracted = tensor.extract %5[%9, %10] : tensor<?x512xi32>
          %12 = arith.index_cast %extracted : i32 to index
          %13 = arith.subi %0, %c1 : index
          %14 = arith.maxsi %12, %c0 : index
          %15 = arith.minsi %14, %13 : index
          %extracted_0 = tensor.extract %6[%15, %11] : tensor<?x768xf32>
          linalg.yield %extracted_0 : f32
        } -> tensor<?x512x768xf32>
        flow.dispatch.tensor.store %8, %4, offsets = [0, 0, 0], sizes = [%1, 512, 768], strides = [1, 1, 1] : tensor<?x512x768xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x512x768xf32>>{%1}
        return
      }
    }
  }
  func.func @f_4_dynamic_Nx512_Vx768_Nx512x768(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<?x768xf32>{%0}
    %2 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
    %3 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<?x512xi32>{%2}
    %4 = flow.dispatch @f_4_dynamic_Nx512_Vx768_Nx512x768_dispatch_0::@f_4_dynamic_Nx512_Vx768_Nx512x768_dispatch_0_generic_Dx512x768_f32[%0, %2](%3, %1, %0, %2) : (tensor<?x512xi32>{%2}, tensor<?x768xf32>{%0}, index, index) -> tensor<?x512x768xf32>{%2}
    %5 = hal.tensor.export %4 "output 0" : tensor<?x512x768xf32>{%2} -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
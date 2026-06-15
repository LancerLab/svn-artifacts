module @f_4_dynamic_Nx256x56x56_Nx56x56x256 {
  flow.executable private @f_4_dynamic_Nx256x56x56_Nx56x56x256_dispatch_0 {
    flow.executable.export public @f_4_dynamic_Nx256x56x56_Nx56x56x256_dispatch_0_generic_Dx56x56x256_f32 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_4_dynamic_Nx256x56x56_Nx56x56x256_dispatch_0_generic_Dx56x56x256_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<?x256x56x56xf32>>, %arg1: index, %arg2: !flow.dispatch.tensor<writeonly:tensor<?x56x56x256xf32>>) {
        %0 = flow.dispatch.workload.ordinal %arg1, 0 : index
        %1 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x256x56x56xf32>>{%0}
        %2 = flow.dispatch.tie_shape %arg2 : !flow.dispatch.tensor<writeonly:tensor<?x56x56x256xf32>>{%0}
        %3 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [%0, 256, 56, 56], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x256x56x56xf32>>{%0} -> tensor<?x256x56x56xf32>
        %4 = tensor.empty(%0) : tensor<?x56x56x256xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<?x256x56x56xf32>) outs(%4 : tensor<?x56x56x256xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<?x56x56x256xf32>
        flow.dispatch.tensor.store %5, %2, offsets = [0, 0, 0, 0], sizes = [%0, 56, 56, 256], strides = [1, 1, 1, 1] : tensor<?x56x56x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x56x56x256xf32>>{%0}
        return
      }
    }
  }
  func.func @f_4_dynamic_Nx256x56x56_Nx56x56x256(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<?x256x56x56xf32>{%0}
    %2 = flow.dispatch @f_4_dynamic_Nx256x56x56_Nx56x56x256_dispatch_0::@f_4_dynamic_Nx256x56x56_Nx56x56x256_dispatch_0_generic_Dx56x56x256_f32[%0](%1, %0) : (tensor<?x256x56x56xf32>{%0}, index) -> tensor<?x56x56x256xf32>{%0}
    %3 = hal.tensor.export %2 "output 0" : tensor<?x56x56x256xf32>{%0} -> !hal.buffer_view
    return %3 : !hal.buffer_view
  }
}
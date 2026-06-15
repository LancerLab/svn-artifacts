module @f_4_dynamic_Nx256x56x56_256_256_Nx256x56x56 {
  util.global private @hoisted = dense<1.00000501> : tensor<256xf32>
  flow.executable private @f_4_dynamic_Nx256x56x56_256_256_Nx256x56x56_dispatch_0 {
    flow.executable.export public @f_4_dynamic_Nx256x56x56_256_256_Nx256x56x56_dispatch_0_generic_Dx256x56x56_f32 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_4_dynamic_Nx256x56x56_256_256_Nx256x56x56_dispatch_0_generic_Dx256x56x56_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<?x256x56x56xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<256xf32>>, %arg2: !flow.dispatch.tensor<readonly:tensor<256xf32>>, %arg3: !flow.dispatch.tensor<readonly:tensor<256xf32>>, %arg4: index, %arg5: !flow.dispatch.tensor<writeonly:tensor<?x256x56x56xf32>>) {
        %0 = flow.dispatch.workload.ordinal %arg4, 0 : index
        %1 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x256x56x56xf32>>{%0}
        %2 = flow.dispatch.tie_shape %arg5 : !flow.dispatch.tensor<writeonly:tensor<?x256x56x56xf32>>{%0}
        %3 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [%0, 256, 56, 56], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x256x56x56xf32>>{%0} -> tensor<?x256x56x56xf32>
        %4 = flow.dispatch.tensor.load %arg1, offsets = [0], sizes = [256], strides = [1] : !flow.dispatch.tensor<readonly:tensor<256xf32>> -> tensor<256xf32>
        %5 = flow.dispatch.tensor.load %arg2, offsets = [0], sizes = [256], strides = [1] : !flow.dispatch.tensor<readonly:tensor<256xf32>> -> tensor<256xf32>
        %6 = flow.dispatch.tensor.load %arg3, offsets = [0], sizes = [256], strides = [1] : !flow.dispatch.tensor<readonly:tensor<256xf32>> -> tensor<256xf32>
        %7 = tensor.empty(%0) : tensor<?x256x56x56xf32>
        %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %4, %5, %6 : tensor<?x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%7 : tensor<?x256x56x56xf32>) {
        ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
          %9 = arith.mulf %in, %in_0 : f32
          %10 = arith.divf %9, %in_1 : f32
          %11 = arith.addf %10, %in_2 : f32
          linalg.yield %11 : f32
        } -> tensor<?x256x56x56xf32>
        flow.dispatch.tensor.store %8, %2, offsets = [0, 0, 0, 0], sizes = [%0, 256, 56, 56], strides = [1, 1, 1, 1] : tensor<?x256x56x56xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x256x56x56xf32>>{%0}
        return
      }
    }
  }
  func.func @f_4_dynamic_Nx256x56x56_256_256_Nx256x56x56(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<?x256x56x56xf32>{%0}
    %2 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<256xf32>
    %3 = hal.tensor.import %arg2 "input 2" : !hal.buffer_view -> tensor<256xf32>
    %hoisted = util.global.load @hoisted : tensor<256xf32>
    %4 = flow.dispatch @f_4_dynamic_Nx256x56x56_256_256_Nx256x56x56_dispatch_0::@f_4_dynamic_Nx256x56x56_256_256_Nx256x56x56_dispatch_0_generic_Dx256x56x56_f32[%0](%1, %2, %hoisted, %3, %0) : (tensor<?x256x56x56xf32>{%0}, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, index) -> tensor<?x256x56x56xf32>{%0}
    %5 = hal.tensor.export %4 "output 0" : tensor<?x256x56x56xf32>{%0} -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
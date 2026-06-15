module @f_3_densenet_32x128x56x56_32x128x28x28 {
  flow.executable private @f_3_densenet_32x128x56x56_32x128x28x28_dispatch_0 {
    flow.executable.export public @f_3_densenet_32x128x56x56_32x128x28x28_dispatch_0_generic_32x128x28x28x2x2_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_3_densenet_32x128x56x56_32x128x28x28_dispatch_0_generic_32x128x28x28x2x2_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<32x128x56x56xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<32x128x28x28xf32>>) {
        %cst = arith.constant -3.402820e+38 : f32
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [32, 128, 56, 56], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x128x56x56xf32>> -> tensor<32x128x56x56xf32>
        %1 = tensor.empty() : tensor<2x2xf32>
        %2 = tensor.empty() : tensor<32x128x28x28xf32>
        %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2 : tensor<32x128x28x28xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        } -> tensor<32x128x28x28xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d4, d3 * 2 + d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %1 : tensor<32x128x56x56xf32>, tensor<2x2xf32>) outs(%3 : tensor<32x128x28x28xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %5 = arith.maxf %out, %in : f32
          linalg.yield %5 : f32
        } -> tensor<32x128x28x28xf32>
        flow.dispatch.tensor.store %4, %arg1, offsets = [0, 0, 0, 0], sizes = [32, 128, 28, 28], strides = [1, 1, 1, 1] : tensor<32x128x28x28xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x128x28x28xf32>>
        return
      }
    }
  }
  func.func @f_3_densenet_32x128x56x56_32x128x28x28(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<32x128x56x56xf32>
    %1 = flow.dispatch @f_3_densenet_32x128x56x56_32x128x28x28_dispatch_0::@f_3_densenet_32x128x56x56_32x128x28x28_dispatch_0_generic_32x128x28x28x2x2_f32(%0) : (tensor<32x128x56x56xf32>) -> tensor<32x128x28x28xf32>
    %2 = hal.tensor.export %1 "output 0" : tensor<32x128x28x28xf32> -> !hal.buffer_view
    return %2 : !hal.buffer_view
  }
}
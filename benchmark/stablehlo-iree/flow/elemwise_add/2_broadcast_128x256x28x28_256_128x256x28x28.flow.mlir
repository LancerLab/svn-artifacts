module @f_2_broadcast_128x256x28x28_256_128x256x28x28 {
  flow.executable private @f_2_broadcast_128x256x28x28_256_128x256x28x28_dispatch_0 {
    flow.executable.export public @f_2_broadcast_128x256x28x28_256_128x256x28x28_dispatch_0_generic_128x256x28x28_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_2_broadcast_128x256x28x28_256_128x256x28x28_dispatch_0_generic_128x256x28x28_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<128x256x28x28xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<256xf32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<128x256x28x28xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [128, 256, 28, 28], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256x28x28xf32>> -> tensor<128x256x28x28xf32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0], sizes = [256], strides = [1] : !flow.dispatch.tensor<readonly:tensor<256xf32>> -> tensor<256xf32>
        %2 = tensor.empty() : tensor<128x256x28x28xf32>
        %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0, %1 : tensor<128x256x28x28xf32>, tensor<256xf32>) outs(%2 : tensor<128x256x28x28xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %4 = arith.addf %in, %in_0 : f32
          linalg.yield %4 : f32
        } -> tensor<128x256x28x28xf32>
        flow.dispatch.tensor.store %3, %arg2, offsets = [0, 0, 0, 0], sizes = [128, 256, 28, 28], strides = [1, 1, 1, 1] : tensor<128x256x28x28xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x256x28x28xf32>>
        return
      }
    }
  }
  func.func @f_2_broadcast_128x256x28x28_256_128x256x28x28(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<128x256x28x28xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<256xf32>
    %2 = flow.dispatch @f_2_broadcast_128x256x28x28_256_128x256x28x28_dispatch_0::@f_2_broadcast_128x256x28x28_256_128x256x28x28_dispatch_0_generic_128x256x28x28_f32(%0, %1) : (tensor<128x256x28x28xf32>, tensor<256xf32>) -> tensor<128x256x28x28xf32>
    %3 = hal.tensor.export %2 "output 0" : tensor<128x256x28x28xf32> -> !hal.buffer_view
    return %3 : !hal.buffer_view
  }
}
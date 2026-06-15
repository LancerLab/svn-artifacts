module @f_19_transformer_32x512x2048_32x512x2048_32x512x2048 {
  flow.executable private @f_19_transformer_32x512x2048_32x512x2048_32x512x2048_dispatch_0 {
    flow.executable.export public @f_19_transformer_32x512x2048_32x512x2048_32x512x2048_dispatch_0_generic_33554432_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_19_transformer_32x512x2048_32x512x2048_32x512x2048_dispatch_0_generic_33554432_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<33554432xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<33554432xf32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<33554432xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0], sizes = [33554432], strides = [1] : !flow.dispatch.tensor<readonly:tensor<33554432xf32>> -> tensor<33554432xf32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0], sizes = [33554432], strides = [1] : !flow.dispatch.tensor<readonly:tensor<33554432xf32>> -> tensor<33554432xf32>
        %2 = tensor.empty() : tensor<33554432xf32>
        %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : tensor<33554432xf32>, tensor<33554432xf32>) outs(%2 : tensor<33554432xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %4 = arith.addf %in, %in_0 : f32
          linalg.yield %4 : f32
        } -> tensor<33554432xf32>
        flow.dispatch.tensor.store %3, %arg2, offsets = [0], sizes = [33554432], strides = [1] : tensor<33554432xf32> -> !flow.dispatch.tensor<writeonly:tensor<33554432xf32>>
        return
      }
    }
  }
  func.func @f_19_transformer_32x512x2048_32x512x2048_32x512x2048(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<32x512x2048xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<32x512x2048xf32>
    %2 = flow.tensor.reshape %0 : tensor<32x512x2048xf32> -> tensor<33554432xf32>
    %3 = flow.tensor.reshape %1 : tensor<32x512x2048xf32> -> tensor<33554432xf32>
    %4 = flow.dispatch @f_19_transformer_32x512x2048_32x512x2048_32x512x2048_dispatch_0::@f_19_transformer_32x512x2048_32x512x2048_32x512x2048_dispatch_0_generic_33554432_f32(%2, %3) : (tensor<33554432xf32>, tensor<33554432xf32>) -> tensor<33554432xf32>
    %5 = flow.tensor.reshape %4 : tensor<33554432xf32> -> tensor<32x512x2048xf32>
    %6 = hal.tensor.export %5 "output 0" : tensor<32x512x2048xf32> -> !hal.buffer_view
    return %6 : !hal.buffer_view
  }
}
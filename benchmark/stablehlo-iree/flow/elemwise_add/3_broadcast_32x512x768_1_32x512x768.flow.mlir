module @f_3_broadcast_32x512x768_1_32x512x768 {
  flow.executable private @f_3_broadcast_32x512x768_1_32x512x768_dispatch_0 {
    flow.executable.export public @f_3_broadcast_32x512x768_1_32x512x768_dispatch_0_generic_32x512x768_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_3_broadcast_32x512x768_1_32x512x768_dispatch_0_generic_32x512x768_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<32x512x768xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<f32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<32x512x768xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 512, 768], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x512x768xf32>> -> tensor<32x512x768xf32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<f32>> -> tensor<f32>
        %2 = tensor.empty() : tensor<32x512x768xf32>
        %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %1 : tensor<32x512x768xf32>, tensor<f32>) outs(%2 : tensor<32x512x768xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %4 = arith.addf %in, %in_0 : f32
          linalg.yield %4 : f32
        } -> tensor<32x512x768xf32>
        flow.dispatch.tensor.store %3, %arg2, offsets = [0, 0, 0], sizes = [32, 512, 768], strides = [1, 1, 1] : tensor<32x512x768xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x512x768xf32>>
        return
      }
    }
  }
  func.func @f_3_broadcast_32x512x768_1_32x512x768(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<32x512x768xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<1xf32>
    %2 = flow.tensor.reshape %1 : tensor<1xf32> -> tensor<f32>
    %3 = flow.dispatch @f_3_broadcast_32x512x768_1_32x512x768_dispatch_0::@f_3_broadcast_32x512x768_1_32x512x768_dispatch_0_generic_32x512x768_f32(%0, %2) : (tensor<32x512x768xf32>, tensor<f32>) -> tensor<32x512x768xf32>
    %4 = hal.tensor.export %3 "output 0" : tensor<32x512x768xf32> -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
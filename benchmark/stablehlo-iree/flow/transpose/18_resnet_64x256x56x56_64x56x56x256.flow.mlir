module @f_18_resnet_64x256x56x56_64x56x56x256 {
  flow.executable private @f_18_resnet_64x256x56x56_64x56x56x256_dispatch_0 {
    flow.executable.export public @f_18_resnet_64x256x56x56_64x56x56x256_dispatch_0_generic_64x3136x256_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_18_resnet_64x256x56x56_64x56x56x256_dispatch_0_generic_64x3136x256_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<64x256x3136xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<64x3136x256xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [64, 256, 3136], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x256x3136xf32>> -> tensor<64x256x3136xf32>
        %1 = tensor.empty() : tensor<64x3136x256xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<64x256x3136xf32>) outs(%1 : tensor<64x3136x256xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<64x3136x256xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [64, 3136, 256], strides = [1, 1, 1] : tensor<64x3136x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x3136x256xf32>>
        return
      }
    }
  }
  func.func @f_18_resnet_64x256x56x56_64x56x56x256(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<64x256x56x56xf32>
    %1 = flow.tensor.reshape %0 : tensor<64x256x56x56xf32> -> tensor<64x256x3136xf32>
    %2 = flow.dispatch @f_18_resnet_64x256x56x56_64x56x56x256_dispatch_0::@f_18_resnet_64x256x56x56_64x56x56x256_dispatch_0_generic_64x3136x256_f32(%1) : (tensor<64x256x3136xf32>) -> tensor<64x3136x256xf32>
    %3 = flow.tensor.reshape %2 : tensor<64x3136x256xf32> -> tensor<64x56x56x256xf32>
    %4 = hal.tensor.export %3 "output 0" : tensor<64x56x56x256xf32> -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
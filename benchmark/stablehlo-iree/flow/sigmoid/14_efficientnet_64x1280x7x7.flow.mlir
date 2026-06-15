module @f_14_efficientnet_64x1280x7x7 {
  flow.executable private @f_14_efficientnet_64x1280x7x7_dispatch_0 {
    flow.executable.export public @f_14_efficientnet_64x1280x7x7_dispatch_0_generic_4014080_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_14_efficientnet_64x1280x7x7_dispatch_0_generic_4014080_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<4014080xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<4014080xf32>>) {
        %cst = arith.constant 1.000000e+00 : f32
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0], sizes = [4014080], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4014080xf32>> -> tensor<4014080xf32>
        %1 = tensor.empty() : tensor<4014080xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0 : tensor<4014080xf32>) outs(%1 : tensor<4014080xf32>) {
        ^bb0(%in: f32, %out: f32):
          %3 = arith.negf %in : f32
          %4 = math.exp %3 : f32
          %5 = arith.addf %4, %cst : f32
          %6 = arith.divf %cst, %5 : f32
          linalg.yield %6 : f32
        } -> tensor<4014080xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [4014080], strides = [1] : tensor<4014080xf32> -> !flow.dispatch.tensor<writeonly:tensor<4014080xf32>>
        return
      }
    }
  }
  func.func @f_14_efficientnet_64x1280x7x7(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<64x1280x7x7xf32>
    %1 = flow.tensor.reshape %0 : tensor<64x1280x7x7xf32> -> tensor<4014080xf32>
    %2 = flow.dispatch @f_14_efficientnet_64x1280x7x7_dispatch_0::@f_14_efficientnet_64x1280x7x7_dispatch_0_generic_4014080_f32(%1) : (tensor<4014080xf32>) -> tensor<4014080xf32>
    %3 = flow.tensor.reshape %2 : tensor<4014080xf32> -> tensor<64x1280x7x7xf32>
    %4 = hal.tensor.export %3 "output 0" : tensor<64x1280x7x7xf32> -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
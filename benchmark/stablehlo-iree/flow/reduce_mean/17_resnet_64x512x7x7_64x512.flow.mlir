module @f_17_resnet_64x512x7x7_64x512 {
  flow.executable private @f_17_resnet_64x512x7x7_64x512_dispatch_0 {
    flow.executable.export public @f_17_resnet_64x512x7x7_64x512_dispatch_0_generic_64x512x49_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_17_resnet_64x512x7x7_64x512_dispatch_0_generic_64x512x49_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<64x512x49xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<64x512xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 4.900000e+01 : f32
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [64, 512, 49], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x512x49xf32>> -> tensor<64x512x49xf32>
        %1 = tensor.empty() : tensor<64x512xf32>
        %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<64x512xf32>) -> tensor<64x512xf32>
        %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0 : tensor<64x512x49xf32>) outs(%2 : tensor<64x512xf32>) {
        ^bb0(%in: f32, %out: f32):
          %5 = arith.addf %out, %in : f32
          linalg.yield %5 : f32
        } -> tensor<64x512xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<64x512xf32>) outs(%1 : tensor<64x512xf32>) {
        ^bb0(%in: f32, %out: f32):
          %5 = arith.divf %in, %cst_0 : f32
          linalg.yield %5 : f32
        } -> tensor<64x512xf32>
        flow.dispatch.tensor.store %4, %arg1, offsets = [0, 0], sizes = [64, 512], strides = [1, 1] : tensor<64x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x512xf32>>
        return
      }
    }
  }
  func.func @f_17_resnet_64x512x7x7_64x512(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<64x512x7x7xf32>
    %1 = flow.tensor.reshape %0 : tensor<64x512x7x7xf32> -> tensor<64x512x49xf32>
    %2 = flow.dispatch @f_17_resnet_64x512x7x7_64x512_dispatch_0::@f_17_resnet_64x512x7x7_64x512_dispatch_0_generic_64x512x49_f32(%1) : (tensor<64x512x49xf32>) -> tensor<64x512xf32>
    %3 = hal.tensor.export %2 "output 0" : tensor<64x512xf32> -> !hal.buffer_view
    return %3 : !hal.buffer_view
  }
}
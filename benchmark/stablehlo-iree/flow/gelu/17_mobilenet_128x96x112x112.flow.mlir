module @f_17_mobilenet_128x96x112x112 {
  flow.executable private @f_17_mobilenet_128x96x112x112_dispatch_0 {
    flow.executable.export public @f_17_mobilenet_128x96x112x112_dispatch_0_generic_154140672_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_17_mobilenet_128x96x112x112_dispatch_0_generic_154140672_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<154140672xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<154140672xf32>>) {
        %cst = arith.constant 4.471500e-02 : f32
        %cst_0 = arith.constant 0.797884583 : f32
        %cst_1 = arith.constant 1.000000e+00 : f32
        %cst_2 = arith.constant 5.000000e-01 : f32
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0], sizes = [154140672], strides = [1] : !flow.dispatch.tensor<readonly:tensor<154140672xf32>> -> tensor<154140672xf32>
        %1 = tensor.empty() : tensor<154140672xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0 : tensor<154140672xf32>) outs(%1 : tensor<154140672xf32>) {
        ^bb0(%in: f32, %out: f32):
          %3 = arith.mulf %in, %in : f32
          %4 = arith.mulf %3, %in : f32
          %5 = arith.mulf %4, %cst : f32
          %6 = arith.addf %in, %5 : f32
          %7 = arith.mulf %6, %cst_0 : f32
          %8 = math.tanh %7 : f32
          %9 = arith.addf %8, %cst_1 : f32
          %10 = arith.mulf %in, %cst_2 : f32
          %11 = arith.mulf %10, %9 : f32
          linalg.yield %11 : f32
        } -> tensor<154140672xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [154140672], strides = [1] : tensor<154140672xf32> -> !flow.dispatch.tensor<writeonly:tensor<154140672xf32>>
        return
      }
    }
  }
  func.func @f_17_mobilenet_128x96x112x112(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<128x96x112x112xf32>
    %1 = flow.tensor.reshape %0 : tensor<128x96x112x112xf32> -> tensor<154140672xf32>
    %2 = flow.dispatch @f_17_mobilenet_128x96x112x112_dispatch_0::@f_17_mobilenet_128x96x112x112_dispatch_0_generic_154140672_f32(%1) : (tensor<154140672xf32>) -> tensor<154140672xf32>
    %3 = flow.tensor.reshape %2 : tensor<154140672xf32> -> tensor<128x96x112x112xf32>
    %4 = hal.tensor.export %3 "output 0" : tensor<128x96x112x112xf32> -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
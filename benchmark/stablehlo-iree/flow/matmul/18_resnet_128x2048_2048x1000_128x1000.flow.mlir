module @f_18_resnet_128x2048_2048x1000_128x1000 {
  flow.executable private @f_18_resnet_128x2048_2048x1000_128x1000_dispatch_0 {
    flow.executable.export public @f_18_resnet_128x2048_2048x1000_128x1000_dispatch_0_matmul_128x1000x2048_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_18_resnet_128x2048_2048x1000_128x1000_dispatch_0_matmul_128x1000x2048_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<128x2048xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<2048x1000xf32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<128x1000xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [128, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x2048xf32>> -> tensor<128x2048xf32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2048, 1000], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1000xf32>> -> tensor<2048x1000xf32>
        %2 = tensor.empty() : tensor<128x1000xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128x1000xf32>) -> tensor<128x1000xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<128x2048xf32>, tensor<2048x1000xf32>) outs(%3 : tensor<128x1000xf32>) -> tensor<128x1000xf32>
        flow.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [128, 1000], strides = [1, 1] : tensor<128x1000xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x1000xf32>>
        return
      }
    }
  }
  func.func @f_18_resnet_128x2048_2048x1000_128x1000(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<128x2048xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<2048x1000xf32>
    %2 = flow.dispatch @f_18_resnet_128x2048_2048x1000_128x1000_dispatch_0::@f_18_resnet_128x2048_2048x1000_128x1000_dispatch_0_matmul_128x1000x2048_f32(%0, %1) : (tensor<128x2048xf32>, tensor<2048x1000xf32>) -> tensor<128x1000xf32>
    %3 = hal.tensor.export %2 "output 0" : tensor<128x1000xf32> -> !hal.buffer_view
    return %3 : !hal.buffer_view
  }
}
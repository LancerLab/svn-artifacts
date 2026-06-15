module @f_15_gpt_16x1024x1536_1536x6144_16x1024x6144 {
  flow.executable private @f_15_gpt_16x1024x1536_1536x6144_16x1024x6144_dispatch_0 {
    flow.executable.export public @f_15_gpt_16x1024x1536_1536x6144_16x1024x6144_dispatch_0_matmul_16384x6144x1536_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_15_gpt_16x1024x1536_1536x6144_16x1024x6144_dispatch_0_matmul_16384x6144x1536_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<16384x1536xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<1536x6144xf32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<16384x6144xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [16384, 1536], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16384x1536xf32>> -> tensor<16384x1536xf32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [1536, 6144], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1536x6144xf32>> -> tensor<1536x6144xf32>
        %2 = tensor.empty() : tensor<16384x6144xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<16384x6144xf32>) -> tensor<16384x6144xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<16384x1536xf32>, tensor<1536x6144xf32>) outs(%3 : tensor<16384x6144xf32>) -> tensor<16384x6144xf32>
        flow.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [16384, 6144], strides = [1, 1] : tensor<16384x6144xf32> -> !flow.dispatch.tensor<writeonly:tensor<16384x6144xf32>>
        return
      }
    }
  }
  func.func @f_15_gpt_16x1024x1536_1536x6144_16x1024x6144(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<16x1024x1536xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<1536x6144xf32>
    %2 = flow.tensor.reshape %0 : tensor<16x1024x1536xf32> -> tensor<16384x1536xf32>
    %3 = flow.dispatch @f_15_gpt_16x1024x1536_1536x6144_16x1024x6144_dispatch_0::@f_15_gpt_16x1024x1536_1536x6144_16x1024x6144_dispatch_0_matmul_16384x6144x1536_f32(%2, %1) : (tensor<16384x1536xf32>, tensor<1536x6144xf32>) -> tensor<16384x6144xf32>
    %4 = flow.tensor.reshape %3 : tensor<16384x6144xf32> -> tensor<16x1024x6144xf32>
    %5 = hal.tensor.export %4 "output 0" : tensor<16x1024x6144xf32> -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
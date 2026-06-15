module @f_19_transformer_64x512x512_512x2048_64x512x2048 {
  flow.executable private @f_19_transformer_64x512x512_512x2048_64x512x2048_dispatch_0 {
    flow.executable.export public @f_19_transformer_64x512x512_512x2048_64x512x2048_dispatch_0_matmul_32768x2048x512_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_19_transformer_64x512x512_512x2048_64x512x2048_dispatch_0_matmul_32768x2048x512_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<32768x512xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<512x2048xf32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<32768x2048xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [32768, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32768x512xf32>> -> tensor<32768x512xf32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [512, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x2048xf32>> -> tensor<512x2048xf32>
        %2 = tensor.empty() : tensor<32768x2048xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32768x2048xf32>) -> tensor<32768x2048xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<32768x512xf32>, tensor<512x2048xf32>) outs(%3 : tensor<32768x2048xf32>) -> tensor<32768x2048xf32>
        flow.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [32768, 2048], strides = [1, 1] : tensor<32768x2048xf32> -> !flow.dispatch.tensor<writeonly:tensor<32768x2048xf32>>
        return
      }
    }
  }
  func.func @f_19_transformer_64x512x512_512x2048_64x512x2048(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<64x512x512xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<512x2048xf32>
    %2 = flow.tensor.reshape %0 : tensor<64x512x512xf32> -> tensor<32768x512xf32>
    %3 = flow.dispatch @f_19_transformer_64x512x512_512x2048_64x512x2048_dispatch_0::@f_19_transformer_64x512x512_512x2048_64x512x2048_dispatch_0_matmul_32768x2048x512_f32(%2, %1) : (tensor<32768x512xf32>, tensor<512x2048xf32>) -> tensor<32768x2048xf32>
    %4 = flow.tensor.reshape %3 : tensor<32768x2048xf32> -> tensor<64x512x2048xf32>
    %5 = hal.tensor.export %4 "output 0" : tensor<64x512x2048xf32> -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
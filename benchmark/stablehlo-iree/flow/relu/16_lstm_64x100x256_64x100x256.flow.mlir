module @f_16_lstm_64x100x256_64x100x256 {
  flow.executable private @f_16_lstm_64x100x256_64x100x256_dispatch_0 {
    flow.executable.export public @f_16_lstm_64x100x256_64x100x256_dispatch_0_generic_1638400_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_16_lstm_64x100x256_64x100x256_dispatch_0_generic_1638400_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<1638400xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<1638400xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 3.40282347E+38 : f32
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0], sizes = [1638400], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1638400xf32>> -> tensor<1638400xf32>
        %1 = tensor.empty() : tensor<1638400xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0 : tensor<1638400xf32>) outs(%1 : tensor<1638400xf32>) {
        ^bb0(%in: f32, %out: f32):
          %3 = arith.maxf %in, %cst : f32
          %4 = arith.minf %3, %cst_0 : f32
          linalg.yield %4 : f32
        } -> tensor<1638400xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [1638400], strides = [1] : tensor<1638400xf32> -> !flow.dispatch.tensor<writeonly:tensor<1638400xf32>>
        return
      }
    }
  }
  func.func @f_16_lstm_64x100x256_64x100x256(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<64x100x256xf32>
    %1 = flow.tensor.reshape %0 : tensor<64x100x256xf32> -> tensor<1638400xf32>
    %2 = flow.dispatch @f_16_lstm_64x100x256_64x100x256_dispatch_0::@f_16_lstm_64x100x256_64x100x256_dispatch_0_generic_1638400_f32(%1) : (tensor<1638400xf32>) -> tensor<1638400xf32>
    %3 = flow.tensor.reshape %2 : tensor<1638400xf32> -> tensor<64x100x256xf32>
    %4 = hal.tensor.export %3 "output 0" : tensor<64x100x256xf32> -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
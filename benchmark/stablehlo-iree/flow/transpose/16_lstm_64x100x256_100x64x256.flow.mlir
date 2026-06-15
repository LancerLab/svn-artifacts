module @f_16_lstm_64x100x256_100x64x256 {
  flow.executable private @f_16_lstm_64x100x256_100x64x256_dispatch_0 {
    flow.executable.export public @f_16_lstm_64x100x256_100x64x256_dispatch_0_generic_100x64x256_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_16_lstm_64x100x256_100x64x256_dispatch_0_generic_100x64x256_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<64x100x256xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<100x64x256xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [64, 100, 256], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x100x256xf32>> -> tensor<64x100x256xf32>
        %1 = tensor.empty() : tensor<100x64x256xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<64x100x256xf32>) outs(%1 : tensor<100x64x256xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<100x64x256xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [100, 64, 256], strides = [1, 1, 1] : tensor<100x64x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<100x64x256xf32>>
        return
      }
    }
  }
  func.func @f_16_lstm_64x100x256_100x64x256(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<64x100x256xf32>
    %1 = flow.dispatch @f_16_lstm_64x100x256_100x64x256_dispatch_0::@f_16_lstm_64x100x256_100x64x256_dispatch_0_generic_100x64x256_f32(%0) : (tensor<64x100x256xf32>) -> tensor<100x64x256xf32>
    %2 = hal.tensor.export %1 "output 0" : tensor<100x64x256xf32> -> !hal.buffer_view
    return %2 : !hal.buffer_view
  }
}
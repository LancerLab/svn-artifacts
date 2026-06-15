module @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28 {
  flow.executable private @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_dispatch_0 {
    flow.executable.export public @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_dispatch_0 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_dispatch_0(%arg0: !flow.dispatch.tensor<readonly:tensor<128x128x28x28xf32>>, %arg1: !flow.dispatch.tensor<readwrite:tensor<128x384x28x28xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [128, 128, 28, 28], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x128x28x28xf32>> -> tensor<128x128x28x28xf32>
        flow.dispatch.tensor.store %0, %arg1, offsets = [0, 0, 0, 0], sizes = [128, 128, 28, 28], strides = [1, 1, 1, 1] : tensor<128x128x28x28xf32> -> !flow.dispatch.tensor<readwrite:tensor<128x384x28x28xf32>>
        return
      }
    }
  }
  flow.executable private @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_dispatch_1 {
    flow.executable.export public @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_dispatch_1 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_dispatch_1(%arg0: !flow.dispatch.tensor<readonly:tensor<128x256x28x28xf32>>, %arg1: !flow.dispatch.tensor<readwrite:tensor<128x384x28x28xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [128, 256, 28, 28], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256x28x28xf32>> -> tensor<128x256x28x28xf32>
        flow.dispatch.tensor.store %0, %arg1, offsets = [0, 128, 0, 0], sizes = [128, 256, 28, 28], strides = [1, 1, 1, 1] : tensor<128x256x28x28xf32> -> !flow.dispatch.tensor<readwrite:tensor<128x384x28x28xf32>>
        return
      }
    }
  }
  func.func @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<128x128x28x28xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<128x256x28x28xf32>
    %2 = flow.tensor.empty : tensor<128x384x28x28xf32>
    %3 = flow.dispatch @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_dispatch_0::@f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_dispatch_0(%0, %2) : (tensor<128x128x28x28xf32>, tensor<128x384x28x28xf32>) -> %2
    %4 = flow.dispatch @f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_dispatch_1::@f_2_cnn_128x128x28x28_128x256x28x28_128x384x28x28_dispatch_1(%1, %3) : (tensor<128x256x28x28xf32>, tensor<128x384x28x28xf32>) -> %3
    %5 = hal.tensor.export %4 "output 0" : tensor<128x384x28x28xf32> -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
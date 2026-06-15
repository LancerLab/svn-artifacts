module @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7 {
  flow.executable private @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_dispatch_0 {
    flow.executable.export public @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_dispatch_0 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_dispatch_0(%arg0: !flow.dispatch.tensor<readonly:tensor<64x1280x7x7xf32>>, %arg1: !flow.dispatch.tensor<readwrite:tensor<64x1600x7x7xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [64, 1280, 7, 7], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x1280x7x7xf32>> -> tensor<64x1280x7x7xf32>
        flow.dispatch.tensor.store %0, %arg1, offsets = [0, 0, 0, 0], sizes = [64, 1280, 7, 7], strides = [1, 1, 1, 1] : tensor<64x1280x7x7xf32> -> !flow.dispatch.tensor<readwrite:tensor<64x1600x7x7xf32>>
        return
      }
    }
  }
  flow.executable private @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_dispatch_1 {
    flow.executable.export public @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_dispatch_1 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_dispatch_1(%arg0: !flow.dispatch.tensor<readonly:tensor<64x320x7x7xf32>>, %arg1: !flow.dispatch.tensor<readwrite:tensor<64x1600x7x7xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [64, 320, 7, 7], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x320x7x7xf32>> -> tensor<64x320x7x7xf32>
        flow.dispatch.tensor.store %0, %arg1, offsets = [0, 1280, 0, 0], sizes = [64, 320, 7, 7], strides = [1, 1, 1, 1] : tensor<64x320x7x7xf32> -> !flow.dispatch.tensor<readwrite:tensor<64x1600x7x7xf32>>
        return
      }
    }
  }
  func.func @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<64x1280x7x7xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<64x320x7x7xf32>
    %2 = flow.tensor.empty : tensor<64x1600x7x7xf32>
    %3 = flow.dispatch @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_dispatch_0::@f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_dispatch_0(%0, %2) : (tensor<64x1280x7x7xf32>, tensor<64x1600x7x7xf32>) -> %2
    %4 = flow.dispatch @f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_dispatch_1::@f_14_efficientnet_64x1280x7x7_64x320x7x7_64x1600x7x7_dispatch_1(%1, %3) : (tensor<64x320x7x7xf32>, tensor<64x1600x7x7xf32>) -> %3
    %5 = hal.tensor.export %4 "output 0" : tensor<64x1600x7x7xf32> -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
module @f_21_vit_32x196x768_32x1x768_32x197x768 {
  flow.executable private @f_21_vit_32x196x768_32x1x768_32x197x768_dispatch_0 {
    flow.executable.export public @f_21_vit_32x196x768_32x1x768_32x197x768_dispatch_0 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_21_vit_32x196x768_32x1x768_32x197x768_dispatch_0(%arg0: !flow.dispatch.tensor<readonly:tensor<32x196x768xf32>>, %arg1: !flow.dispatch.tensor<readwrite:tensor<32x197x768xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 196, 768], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x196x768xf32>> -> tensor<32x196x768xf32>
        flow.dispatch.tensor.store %0, %arg1, offsets = [0, 0, 0], sizes = [32, 196, 768], strides = [1, 1, 1] : tensor<32x196x768xf32> -> !flow.dispatch.tensor<readwrite:tensor<32x197x768xf32>>
        return
      }
    }
  }
  flow.executable private @f_21_vit_32x196x768_32x1x768_32x197x768_dispatch_1 {
    flow.executable.export public @f_21_vit_32x196x768_32x1x768_32x197x768_dispatch_1 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_21_vit_32x196x768_32x1x768_32x197x768_dispatch_1(%arg0: !flow.dispatch.tensor<readonly:tensor<32x768xf32>>, %arg1: !flow.dispatch.tensor<readwrite:tensor<32x197x768xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [32, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x768xf32>> -> tensor<32x768xf32>
        flow.dispatch.tensor.store %0, %arg1, offsets = [0, 196, 0], sizes = [32, 1, 768], strides = [1, 1, 1] : tensor<32x768xf32> -> !flow.dispatch.tensor<readwrite:tensor<32x197x768xf32>>
        return
      }
    }
  }
  func.func @f_21_vit_32x196x768_32x1x768_32x197x768(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<32x196x768xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<32x1x768xf32>
    %2 = flow.tensor.empty : tensor<32x197x768xf32>
    %3 = flow.dispatch @f_21_vit_32x196x768_32x1x768_32x197x768_dispatch_0::@f_21_vit_32x196x768_32x1x768_32x197x768_dispatch_0(%0, %2) : (tensor<32x196x768xf32>, tensor<32x197x768xf32>) -> %2
    %4 = flow.tensor.reshape %1 : tensor<32x1x768xf32> -> tensor<32x768xf32>
    %5 = flow.dispatch @f_21_vit_32x196x768_32x1x768_32x197x768_dispatch_1::@f_21_vit_32x196x768_32x1x768_32x197x768_dispatch_1(%4, %3) : (tensor<32x768xf32>, tensor<32x197x768xf32>) -> %3
    %6 = hal.tensor.export %5 "output 0" : tensor<32x197x768xf32> -> !hal.buffer_view
    return %6 : !hal.buffer_view
  }
}
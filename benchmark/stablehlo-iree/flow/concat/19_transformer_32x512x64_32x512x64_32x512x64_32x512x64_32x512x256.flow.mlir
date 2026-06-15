module @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256 {
  flow.executable private @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_0 {
    flow.executable.export public @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_0 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_0(%arg0: !flow.dispatch.tensor<readonly:tensor<32x512x64xf32>>, %arg1: !flow.dispatch.tensor<readwrite:tensor<32x512x256xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 512, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x512x64xf32>> -> tensor<32x512x64xf32>
        flow.dispatch.tensor.store %0, %arg1, offsets = [0, 0, 0], sizes = [32, 512, 64], strides = [1, 1, 1] : tensor<32x512x64xf32> -> !flow.dispatch.tensor<readwrite:tensor<32x512x256xf32>>
        return
      }
    }
  }
  flow.executable private @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_1 {
    flow.executable.export public @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_1 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_1(%arg0: !flow.dispatch.tensor<readonly:tensor<32x512x64xf32>>, %arg1: !flow.dispatch.tensor<readwrite:tensor<32x512x256xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 512, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x512x64xf32>> -> tensor<32x512x64xf32>
        flow.dispatch.tensor.store %0, %arg1, offsets = [0, 0, 64], sizes = [32, 512, 64], strides = [1, 1, 1] : tensor<32x512x64xf32> -> !flow.dispatch.tensor<readwrite:tensor<32x512x256xf32>>
        return
      }
    }
  }
  flow.executable private @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_2 {
    flow.executable.export public @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_2 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_2(%arg0: !flow.dispatch.tensor<readonly:tensor<32x512x64xf32>>, %arg1: !flow.dispatch.tensor<readwrite:tensor<32x512x256xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 512, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x512x64xf32>> -> tensor<32x512x64xf32>
        flow.dispatch.tensor.store %0, %arg1, offsets = [0, 0, 128], sizes = [32, 512, 64], strides = [1, 1, 1] : tensor<32x512x64xf32> -> !flow.dispatch.tensor<readwrite:tensor<32x512x256xf32>>
        return
      }
    }
  }
  flow.executable private @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_3 {
    flow.executable.export public @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_3 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_3(%arg0: !flow.dispatch.tensor<readonly:tensor<32x512x64xf32>>, %arg1: !flow.dispatch.tensor<readwrite:tensor<32x512x256xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 512, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x512x64xf32>> -> tensor<32x512x64xf32>
        flow.dispatch.tensor.store %0, %arg1, offsets = [0, 0, 192], sizes = [32, 512, 64], strides = [1, 1, 1] : tensor<32x512x64xf32> -> !flow.dispatch.tensor<readwrite:tensor<32x512x256xf32>>
        return
      }
    }
  }
  func.func @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<32x512x64xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<32x512x64xf32>
    %2 = hal.tensor.import %arg2 "input 2" : !hal.buffer_view -> tensor<32x512x64xf32>
    %3 = hal.tensor.import %arg3 "input 3" : !hal.buffer_view -> tensor<32x512x64xf32>
    %4 = flow.tensor.empty : tensor<32x512x256xf32>
    %5 = flow.dispatch @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_0::@f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_0(%0, %4) : (tensor<32x512x64xf32>, tensor<32x512x256xf32>) -> %4
    %6 = flow.dispatch @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_1::@f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_1(%1, %5) : (tensor<32x512x64xf32>, tensor<32x512x256xf32>) -> %5
    %7 = flow.dispatch @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_2::@f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_2(%2, %6) : (tensor<32x512x64xf32>, tensor<32x512x256xf32>) -> %6
    %8 = flow.dispatch @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_3::@f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256_dispatch_3(%3, %7) : (tensor<32x512x64xf32>, tensor<32x512x256xf32>) -> %7
    %9 = hal.tensor.export %8 "output 0" : tensor<32x512x256xf32> -> !hal.buffer_view
    return %9 : !hal.buffer_view
  }
}
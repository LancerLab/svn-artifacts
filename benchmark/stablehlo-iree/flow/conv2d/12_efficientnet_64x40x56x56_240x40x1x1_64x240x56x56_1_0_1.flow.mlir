module @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1 {
  flow.executable private @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_0 {
    flow.executable.export public @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_0_generic_64x3136x40_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_0_generic_64x3136x40_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<64x40x3136xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<64x3136x40xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [64, 40, 3136], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x40x3136xf32>> -> tensor<64x40x3136xf32>
        %1 = tensor.empty() : tensor<64x3136x40xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<64x40x3136xf32>) outs(%1 : tensor<64x3136x40xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<64x3136x40xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [64, 3136, 40], strides = [1, 1, 1] : tensor<64x3136x40xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x3136x40xf32>>
        return
      }
    }
  }
  flow.executable private @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_1 {
    flow.executable.export public @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_1_generic_40x240_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_1_generic_40x240_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<240x40xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<40x240xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [240, 40], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<240x40xf32>> -> tensor<240x40xf32>
        %1 = tensor.empty() : tensor<40x240xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<240x40xf32>) outs(%1 : tensor<40x240xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<40x240xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0], sizes = [40, 240], strides = [1, 1] : tensor<40x240xf32> -> !flow.dispatch.tensor<writeonly:tensor<40x240xf32>>
        return
      }
    }
  }
  flow.executable private @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_2 {
    flow.executable.export public @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_64x56x56x240x1x1x40_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_64x56x56x240x1x1x40_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<64x56x56x40xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<1x1x40x240xf32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<64x56x56x240xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [64, 56, 56, 40], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x56x56x40xf32>> -> tensor<64x56x56x40xf32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [1, 1, 40, 240], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x40x240xf32>> -> tensor<1x1x40x240xf32>
        %2 = tensor.empty() : tensor<64x56x56x240xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<64x56x56x240xf32>) -> tensor<64x56x56x240xf32>
        %4 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%0, %1 : tensor<64x56x56x40xf32>, tensor<1x1x40x240xf32>) outs(%3 : tensor<64x56x56x240xf32>) -> tensor<64x56x56x240xf32>
        flow.dispatch.tensor.store %4, %arg2, offsets = [0, 0, 0, 0], sizes = [64, 56, 56, 240], strides = [1, 1, 1, 1] : tensor<64x56x56x240xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x56x56x240xf32>>
        return
      }
    }
  }
  flow.executable private @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_3 {
    flow.executable.export public @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_3_generic_64x240x3136_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_3_generic_64x240x3136_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<64x3136x240xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<64x240x3136xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [64, 3136, 240], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x3136x240xf32>> -> tensor<64x3136x240xf32>
        %1 = tensor.empty() : tensor<64x240x3136xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<64x3136x240xf32>) outs(%1 : tensor<64x240x3136xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<64x240x3136xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [64, 240, 3136], strides = [1, 1, 1] : tensor<64x240x3136xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x240x3136xf32>>
        return
      }
    }
  }
  func.func @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<64x40x56x56xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<240x40x1x1xf32>
    %2 = flow.tensor.reshape %0 : tensor<64x40x56x56xf32> -> tensor<64x40x3136xf32>
    %3 = flow.dispatch @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_0::@f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_0_generic_64x3136x40_f32(%2) : (tensor<64x40x3136xf32>) -> tensor<64x3136x40xf32>
    %4 = flow.tensor.reshape %3 : tensor<64x3136x40xf32> -> tensor<64x56x56x40xf32>
    %5 = flow.tensor.reshape %1 : tensor<240x40x1x1xf32> -> tensor<240x40xf32>
    %6 = flow.dispatch @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_1::@f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_1_generic_40x240_f32(%5) : (tensor<240x40xf32>) -> tensor<40x240xf32>
    %7 = flow.tensor.reshape %6 : tensor<40x240xf32> -> tensor<1x1x40x240xf32>
    %8 = flow.dispatch @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_2::@f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_64x56x56x240x1x1x40_f32(%4, %7) : (tensor<64x56x56x40xf32>, tensor<1x1x40x240xf32>) -> tensor<64x56x56x240xf32>
    %9 = flow.tensor.reshape %8 : tensor<64x56x56x240xf32> -> tensor<64x3136x240xf32>
    %10 = flow.dispatch @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_3::@f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_dispatch_3_generic_64x240x3136_f32(%9) : (tensor<64x3136x240xf32>) -> tensor<64x240x3136xf32>
    %11 = flow.tensor.reshape %10 : tensor<64x240x3136xf32> -> tensor<64x240x56x56xf32>
    %12 = hal.tensor.export %11 "output 0" : tensor<64x240x56x56xf32> -> !hal.buffer_view
    return %12 : !hal.buffer_view
  }
}
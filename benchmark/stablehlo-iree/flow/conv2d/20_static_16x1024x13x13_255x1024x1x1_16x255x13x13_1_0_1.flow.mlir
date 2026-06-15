module @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1 {
  flow.executable private @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_0 {
    flow.executable.export public @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_0_generic_16x169x1024_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_0_generic_16x169x1024_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<16x1024x169xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<16x169x1024xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [16, 1024, 169], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x1024x169xf32>> -> tensor<16x1024x169xf32>
        %1 = tensor.empty() : tensor<16x169x1024xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<16x1024x169xf32>) outs(%1 : tensor<16x169x1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<16x169x1024xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [16, 169, 1024], strides = [1, 1, 1] : tensor<16x169x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x169x1024xf32>>
        return
      }
    }
  }
  flow.executable private @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_1 {
    flow.executable.export public @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_1_generic_1024x255_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_1_generic_1024x255_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<255x1024xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<1024x255xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [255, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<255x1024xf32>> -> tensor<255x1024xf32>
        %1 = tensor.empty() : tensor<1024x255xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<255x1024xf32>) outs(%1 : tensor<1024x255xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<1024x255xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0], sizes = [1024, 255], strides = [1, 1] : tensor<1024x255xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x255xf32>>
        return
      }
    }
  }
  flow.executable private @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_2 {
    flow.executable.export public @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_16x13x13x255x1x1x1024_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_16x13x13x255x1x1x1024_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<16x13x13x1024xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<1x1x1024x255xf32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<16x13x13x255xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [16, 13, 13, 1024], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x13x13x1024xf32>> -> tensor<16x13x13x1024xf32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [1, 1, 1024, 255], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x1024x255xf32>> -> tensor<1x1x1024x255xf32>
        %2 = tensor.empty() : tensor<16x13x13x255xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<16x13x13x255xf32>) -> tensor<16x13x13x255xf32>
        %4 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%0, %1 : tensor<16x13x13x1024xf32>, tensor<1x1x1024x255xf32>) outs(%3 : tensor<16x13x13x255xf32>) -> tensor<16x13x13x255xf32>
        flow.dispatch.tensor.store %4, %arg2, offsets = [0, 0, 0, 0], sizes = [16, 13, 13, 255], strides = [1, 1, 1, 1] : tensor<16x13x13x255xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x13x13x255xf32>>
        return
      }
    }
  }
  flow.executable private @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_3 {
    flow.executable.export public @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_3_generic_16x255x169_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_3_generic_16x255x169_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<16x169x255xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<16x255x169xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [16, 169, 255], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x169x255xf32>> -> tensor<16x169x255xf32>
        %1 = tensor.empty() : tensor<16x255x169xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<16x169x255xf32>) outs(%1 : tensor<16x255x169xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<16x255x169xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [16, 255, 169], strides = [1, 1, 1] : tensor<16x255x169xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x255x169xf32>>
        return
      }
    }
  }
  func.func @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<16x1024x13x13xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<255x1024x1x1xf32>
    %2 = flow.tensor.reshape %0 : tensor<16x1024x13x13xf32> -> tensor<16x1024x169xf32>
    %3 = flow.dispatch @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_0::@f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_0_generic_16x169x1024_f32(%2) : (tensor<16x1024x169xf32>) -> tensor<16x169x1024xf32>
    %4 = flow.tensor.reshape %3 : tensor<16x169x1024xf32> -> tensor<16x13x13x1024xf32>
    %5 = flow.tensor.reshape %1 : tensor<255x1024x1x1xf32> -> tensor<255x1024xf32>
    %6 = flow.dispatch @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_1::@f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_1_generic_1024x255_f32(%5) : (tensor<255x1024xf32>) -> tensor<1024x255xf32>
    %7 = flow.tensor.reshape %6 : tensor<1024x255xf32> -> tensor<1x1x1024x255xf32>
    %8 = flow.dispatch @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_2::@f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_16x13x13x255x1x1x1024_f32(%4, %7) : (tensor<16x13x13x1024xf32>, tensor<1x1x1024x255xf32>) -> tensor<16x13x13x255xf32>
    %9 = flow.tensor.reshape %8 : tensor<16x13x13x255xf32> -> tensor<16x169x255xf32>
    %10 = flow.dispatch @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_3::@f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_dispatch_3_generic_16x255x169_f32(%9) : (tensor<16x169x255xf32>) -> tensor<16x255x169xf32>
    %11 = flow.tensor.reshape %10 : tensor<16x255x169xf32> -> tensor<16x255x13x13xf32>
    %12 = hal.tensor.export %11 "output 0" : tensor<16x255x13x13xf32> -> !hal.buffer_view
    return %12 : !hal.buffer_view
  }
}
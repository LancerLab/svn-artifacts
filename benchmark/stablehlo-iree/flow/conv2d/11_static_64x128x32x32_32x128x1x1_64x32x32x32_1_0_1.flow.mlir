module @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1 {
  flow.executable private @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_0 {
    flow.executable.export public @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_0_generic_64x1024x128_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_0_generic_64x1024x128_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<64x128x1024xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<64x1024x128xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [64, 128, 1024], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x128x1024xf32>> -> tensor<64x128x1024xf32>
        %1 = tensor.empty() : tensor<64x1024x128xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<64x128x1024xf32>) outs(%1 : tensor<64x1024x128xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<64x1024x128xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [64, 1024, 128], strides = [1, 1, 1] : tensor<64x1024x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x1024x128xf32>>
        return
      }
    }
  }
  flow.executable private @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_1 {
    flow.executable.export public @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_1_generic_128x32_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_1_generic_128x32_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<32x128xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<128x32xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [32, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x128xf32>> -> tensor<32x128xf32>
        %1 = tensor.empty() : tensor<128x32xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<32x128xf32>) outs(%1 : tensor<128x32xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<128x32xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0], sizes = [128, 32], strides = [1, 1] : tensor<128x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x32xf32>>
        return
      }
    }
  }
  flow.executable private @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_2 {
    flow.executable.export public @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_64x32x32x32x1x1x128_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_64x32x32x32x1x1x128_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<64x32x32x128xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<1x1x128x32xf32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<64x32x32x32xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [64, 32, 32, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x32x32x128xf32>> -> tensor<64x32x32x128xf32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [1, 1, 128, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x128x32xf32>> -> tensor<1x1x128x32xf32>
        %2 = tensor.empty() : tensor<64x32x32x32xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<64x32x32x32xf32>) -> tensor<64x32x32x32xf32>
        %4 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%0, %1 : tensor<64x32x32x128xf32>, tensor<1x1x128x32xf32>) outs(%3 : tensor<64x32x32x32xf32>) -> tensor<64x32x32x32xf32>
        flow.dispatch.tensor.store %4, %arg2, offsets = [0, 0, 0, 0], sizes = [64, 32, 32, 32], strides = [1, 1, 1, 1] : tensor<64x32x32x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x32x32x32xf32>>
        return
      }
    }
  }
  flow.executable private @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_3 {
    flow.executable.export public @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_3_generic_64x32x1024_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_3_generic_64x32x1024_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<64x1024x32xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<64x32x1024xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [64, 1024, 32], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x1024x32xf32>> -> tensor<64x1024x32xf32>
        %1 = tensor.empty() : tensor<64x32x1024xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<64x1024x32xf32>) outs(%1 : tensor<64x32x1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<64x32x1024xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [64, 32, 1024], strides = [1, 1, 1] : tensor<64x32x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x32x1024xf32>>
        return
      }
    }
  }
  func.func @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<64x128x32x32xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<32x128x1x1xf32>
    %2 = flow.tensor.reshape %0 : tensor<64x128x32x32xf32> -> tensor<64x128x1024xf32>
    %3 = flow.dispatch @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_0::@f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_0_generic_64x1024x128_f32(%2) : (tensor<64x128x1024xf32>) -> tensor<64x1024x128xf32>
    %4 = flow.tensor.reshape %3 : tensor<64x1024x128xf32> -> tensor<64x32x32x128xf32>
    %5 = flow.tensor.reshape %1 : tensor<32x128x1x1xf32> -> tensor<32x128xf32>
    %6 = flow.dispatch @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_1::@f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_1_generic_128x32_f32(%5) : (tensor<32x128xf32>) -> tensor<128x32xf32>
    %7 = flow.tensor.reshape %6 : tensor<128x32xf32> -> tensor<1x1x128x32xf32>
    %8 = flow.dispatch @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_2::@f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_64x32x32x32x1x1x128_f32(%4, %7) : (tensor<64x32x32x128xf32>, tensor<1x1x128x32xf32>) -> tensor<64x32x32x32xf32>
    %9 = flow.tensor.reshape %8 : tensor<64x32x32x32xf32> -> tensor<64x1024x32xf32>
    %10 = flow.dispatch @f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_3::@f_11_static_64x128x32x32_32x128x1x1_64x32x32x32_1_0_1_dispatch_3_generic_64x32x1024_f32(%9) : (tensor<64x1024x32xf32>) -> tensor<64x32x1024xf32>
    %11 = flow.tensor.reshape %10 : tensor<64x32x1024xf32> -> tensor<64x32x32x32xf32>
    %12 = hal.tensor.export %11 "output 0" : tensor<64x32x32x32xf32> -> !hal.buffer_view
    return %12 : !hal.buffer_view
  }
}
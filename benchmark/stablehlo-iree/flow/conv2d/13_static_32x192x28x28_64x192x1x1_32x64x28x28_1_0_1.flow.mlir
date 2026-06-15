module @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1 {
  flow.executable private @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_dispatch_0 {
    flow.executable.export public @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_dispatch_0_generic_32x784x192_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_dispatch_0_generic_32x784x192_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<32x192x784xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<32x784x192xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 192, 784], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x192x784xf32>> -> tensor<32x192x784xf32>
        %1 = tensor.empty() : tensor<32x784x192xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<32x192x784xf32>) outs(%1 : tensor<32x784x192xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<32x784x192xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [32, 784, 192], strides = [1, 1, 1] : tensor<32x784x192xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x784x192xf32>>
        return
      }
    }
  }
  flow.executable private @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_dispatch_1 {
    flow.executable.export public @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_dispatch_1_generic_192x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_dispatch_1_generic_192x64_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<64x192xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<192x64xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [64, 192], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x192xf32>> -> tensor<64x192xf32>
        %1 = tensor.empty() : tensor<192x64xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<64x192xf32>) outs(%1 : tensor<192x64xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<192x64xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0], sizes = [192, 64], strides = [1, 1] : tensor<192x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<192x64xf32>>
        return
      }
    }
  }
  flow.executable private @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_dispatch_2 {
    flow.executable.export public @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_32x28x28x64x1x1x192_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_32x28x28x64x1x1x192_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<32x28x28x192xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<1x1x192x64xf32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<32x28x28x64xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [32, 28, 28, 192], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x28x28x192xf32>> -> tensor<32x28x28x192xf32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [1, 1, 192, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x192x64xf32>> -> tensor<1x1x192x64xf32>
        %2 = tensor.empty() : tensor<32x28x28x64xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32x28x28x64xf32>) -> tensor<32x28x28x64xf32>
        %4 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%0, %1 : tensor<32x28x28x192xf32>, tensor<1x1x192x64xf32>) outs(%3 : tensor<32x28x28x64xf32>) -> tensor<32x28x28x64xf32>
        flow.dispatch.tensor.store %4, %arg2, offsets = [0, 0, 0, 0], sizes = [32, 28, 28, 64], strides = [1, 1, 1, 1] : tensor<32x28x28x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x28x28x64xf32>>
        return
      }
    }
  }
  flow.executable private @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_dispatch_3 {
    flow.executable.export public @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_dispatch_3_generic_32x64x784_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_dispatch_3_generic_32x64x784_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<32x784x64xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<32x64x784xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 784, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x784x64xf32>> -> tensor<32x784x64xf32>
        %1 = tensor.empty() : tensor<32x64x784xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<32x784x64xf32>) outs(%1 : tensor<32x64x784xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<32x64x784xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [32, 64, 784], strides = [1, 1, 1] : tensor<32x64x784xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x64x784xf32>>
        return
      }
    }
  }
  func.func @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<32x192x28x28xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<64x192x1x1xf32>
    %2 = flow.tensor.reshape %0 : tensor<32x192x28x28xf32> -> tensor<32x192x784xf32>
    %3 = flow.dispatch @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_dispatch_0::@f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_dispatch_0_generic_32x784x192_f32(%2) : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %4 = flow.tensor.reshape %3 : tensor<32x784x192xf32> -> tensor<32x28x28x192xf32>
    %5 = flow.tensor.reshape %1 : tensor<64x192x1x1xf32> -> tensor<64x192xf32>
    %6 = flow.dispatch @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_dispatch_1::@f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_dispatch_1_generic_192x64_f32(%5) : (tensor<64x192xf32>) -> tensor<192x64xf32>
    %7 = flow.tensor.reshape %6 : tensor<192x64xf32> -> tensor<1x1x192x64xf32>
    %8 = flow.dispatch @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_dispatch_2::@f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_dispatch_2_conv_2d_nhwc_hwcf_32x28x28x64x1x1x192_f32(%4, %7) : (tensor<32x28x28x192xf32>, tensor<1x1x192x64xf32>) -> tensor<32x28x28x64xf32>
    %9 = flow.tensor.reshape %8 : tensor<32x28x28x64xf32> -> tensor<32x784x64xf32>
    %10 = flow.dispatch @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_dispatch_3::@f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_dispatch_3_generic_32x64x784_f32(%9) : (tensor<32x784x64xf32>) -> tensor<32x64x784xf32>
    %11 = flow.tensor.reshape %10 : tensor<32x64x784xf32> -> tensor<32x64x28x28xf32>
    %12 = hal.tensor.export %11 "output 0" : tensor<32x64x28x28xf32> -> !hal.buffer_view
    return %12 : !hal.buffer_view
  }
}
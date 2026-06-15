module @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D {
  flow.executable private @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_0 {
    flow.executable.export public @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_0_generic_128x12544x32_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_0_generic_128x12544x32_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<128x32x12544xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<128x12544x32xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [128, 32, 12544], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x32x12544xf32>> -> tensor<128x32x12544xf32>
        %1 = tensor.empty() : tensor<128x12544x32xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<128x32x12544xf32>) outs(%1 : tensor<128x12544x32xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<128x12544x32xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [128, 12544, 32], strides = [1, 1, 1] : tensor<128x12544x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x12544x32xf32>>
        return
      }
    }
  }
  flow.executable private @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_1 {
    flow.executable.export public @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_1_generic_9x32x32_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_1_generic_9x32x32_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<32x32x9xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<9x32x32xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 32, 9], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x32x9xf32>> -> tensor<32x32x9xf32>
        %1 = tensor.empty() : tensor<9x32x32xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<32x32x9xf32>) outs(%1 : tensor<9x32x32xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<9x32x32xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [9, 32, 32], strides = [1, 1, 1] : tensor<9x32x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<9x32x32xf32>>
        return
      }
    }
  }
  flow.executable private @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_2 {
    flow.executable.export public @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_2 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_2(%arg0: !flow.dispatch.tensor<readonly:tensor<128x112x112x32xf32>>, %arg1: !flow.dispatch.tensor<readwrite:tensor<128x114x114x32xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [128, 112, 112, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x112x112x32xf32>> -> tensor<128x112x112x32xf32>
        flow.dispatch.tensor.store %0, %arg1, offsets = [0, 1, 1, 0], sizes = [128, 112, 112, 32], strides = [1, 1, 1, 1] : tensor<128x112x112x32xf32> -> !flow.dispatch.tensor<readwrite:tensor<128x114x114x32xf32>>
        return
      }
    }
  }
  flow.executable private @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_3 {
    flow.executable.export public @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_128x112x112x32x3x3x32_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_128x112x112x32x3x3x32_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<128x114x114x32xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<3x3x32x32xf32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<128x112x112x32xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [128, 114, 114, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x114x114x32xf32>> -> tensor<128x114x114x32xf32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [3, 3, 32, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x32x32xf32>> -> tensor<3x3x32x32xf32>
        %2 = tensor.empty() : tensor<128x112x112x32xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128x112x112x32xf32>) -> tensor<128x112x112x32xf32>
        %4 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%0, %1 : tensor<128x114x114x32xf32>, tensor<3x3x32x32xf32>) outs(%3 : tensor<128x112x112x32xf32>) -> tensor<128x112x112x32xf32>
        flow.dispatch.tensor.store %4, %arg2, offsets = [0, 0, 0, 0], sizes = [128, 112, 112, 32], strides = [1, 1, 1, 1] : tensor<128x112x112x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x112x112x32xf32>>
        return
      }
    }
  }
  flow.executable private @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_4 {
    flow.executable.export public @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_4_generic_128x32x12544_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_4_generic_128x32x12544_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<128x12544x32xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<128x32x12544xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [128, 12544, 32], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x12544x32xf32>> -> tensor<128x12544x32xf32>
        %1 = tensor.empty() : tensor<128x32x12544xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<128x12544x32xf32>) outs(%1 : tensor<128x32x12544xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<128x32x12544xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [128, 32, 12544], strides = [1, 1, 1] : tensor<128x32x12544xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x32x12544xf32>>
        return
      }
    }
  }
  func.func @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<128x32x112x112xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<32x32x3x3xf32>
    %2 = flow.tensor.reshape %0 : tensor<128x32x112x112xf32> -> tensor<128x32x12544xf32>
    %3 = flow.dispatch @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_0::@f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_0_generic_128x12544x32_f32(%2) : (tensor<128x32x12544xf32>) -> tensor<128x12544x32xf32>
    %4 = flow.tensor.reshape %3 : tensor<128x12544x32xf32> -> tensor<128x112x112x32xf32>
    %5 = flow.tensor.reshape %1 : tensor<32x32x3x3xf32> -> tensor<32x32x9xf32>
    %6 = flow.dispatch @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_1::@f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_1_generic_9x32x32_f32(%5) : (tensor<32x32x9xf32>) -> tensor<9x32x32xf32>
    %7 = flow.tensor.reshape %6 : tensor<9x32x32xf32> -> tensor<3x3x32x32xf32>
    %8 = flow.tensor.splat %cst : tensor<128x114x114x32xf32>
    %9 = flow.dispatch @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_2::@f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_2(%4, %8) : (tensor<128x112x112x32xf32>, tensor<128x114x114x32xf32>) -> %8
    %10 = flow.dispatch @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_3::@f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_128x112x112x32x3x3x32_f32(%9, %7) : (tensor<128x114x114x32xf32>, tensor<3x3x32x32xf32>) -> tensor<128x112x112x32xf32>
    %11 = flow.tensor.reshape %10 : tensor<128x112x112x32xf32> -> tensor<128x12544x32xf32>
    %12 = flow.dispatch @f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_4::@f_14_mobilenet_128x32x112x112_32x32x3x3_128x32x112x112_S_P_D_dispatch_4_generic_128x32x12544_f32(%11) : (tensor<128x12544x32xf32>) -> tensor<128x32x12544xf32>
    %13 = flow.tensor.reshape %12 : tensor<128x32x12544xf32> -> tensor<128x32x112x112xf32>
    %14 = hal.tensor.export %13 "output 0" : tensor<128x32x112x112xf32> -> !hal.buffer_view
    return %14 : !hal.buffer_view
  }
}
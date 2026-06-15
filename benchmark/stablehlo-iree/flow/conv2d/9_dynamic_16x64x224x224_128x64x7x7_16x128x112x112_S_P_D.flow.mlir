module @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D {
  flow.executable private @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_0 {
    flow.executable.export public @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_0_generic_16x50176x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_0_generic_16x50176x64_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<16x64x50176xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<16x50176x64xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [16, 64, 50176], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x64x50176xf32>> -> tensor<16x64x50176xf32>
        %1 = tensor.empty() : tensor<16x50176x64xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<16x64x50176xf32>) outs(%1 : tensor<16x50176x64xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<16x50176x64xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [16, 50176, 64], strides = [1, 1, 1] : tensor<16x50176x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x50176x64xf32>>
        return
      }
    }
  }
  flow.executable private @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_1 {
    flow.executable.export public @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_1_generic_49x64x128_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_1_generic_49x64x128_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<128x64x49xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<49x64x128xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [128, 64, 49], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x64x49xf32>> -> tensor<128x64x49xf32>
        %1 = tensor.empty() : tensor<49x64x128xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<128x64x49xf32>) outs(%1 : tensor<49x64x128xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<49x64x128xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [49, 64, 128], strides = [1, 1, 1] : tensor<49x64x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<49x64x128xf32>>
        return
      }
    }
  }
  flow.executable private @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_2 {
    flow.executable.export public @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_2 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_2(%arg0: !flow.dispatch.tensor<readonly:tensor<16x224x224x64xf32>>, %arg1: !flow.dispatch.tensor<readwrite:tensor<16x230x230x64xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [16, 224, 224, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x224x224x64xf32>> -> tensor<16x224x224x64xf32>
        flow.dispatch.tensor.store %0, %arg1, offsets = [0, 3, 3, 0], sizes = [16, 224, 224, 64], strides = [1, 1, 1, 1] : tensor<16x224x224x64xf32> -> !flow.dispatch.tensor<readwrite:tensor<16x230x230x64xf32>>
        return
      }
    }
  }
  flow.executable private @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_3 {
    flow.executable.export public @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_16x112x112x128x7x7x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_16x112x112x128x7x7x64_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<16x230x230x64xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<7x7x64x128xf32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<16x112x112x128xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [16, 230, 230, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x230x230x64xf32>> -> tensor<16x230x230x64xf32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [7, 7, 64, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<7x7x64x128xf32>> -> tensor<7x7x64x128xf32>
        %2 = tensor.empty() : tensor<16x112x112x128xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<16x112x112x128xf32>) -> tensor<16x112x112x128xf32>
        %4 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%0, %1 : tensor<16x230x230x64xf32>, tensor<7x7x64x128xf32>) outs(%3 : tensor<16x112x112x128xf32>) -> tensor<16x112x112x128xf32>
        flow.dispatch.tensor.store %4, %arg2, offsets = [0, 0, 0, 0], sizes = [16, 112, 112, 128], strides = [1, 1, 1, 1] : tensor<16x112x112x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x112x112x128xf32>>
        return
      }
    }
  }
  flow.executable private @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_4 {
    flow.executable.export public @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_4_generic_16x128x12544_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_4_generic_16x128x12544_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<16x12544x128xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<16x128x12544xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [16, 12544, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x12544x128xf32>> -> tensor<16x12544x128xf32>
        %1 = tensor.empty() : tensor<16x128x12544xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<16x12544x128xf32>) outs(%1 : tensor<16x128x12544xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<16x128x12544xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [16, 128, 12544], strides = [1, 1, 1] : tensor<16x128x12544xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x128x12544xf32>>
        return
      }
    }
  }
  func.func @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<16x64x224x224xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<128x64x7x7xf32>
    %2 = flow.tensor.reshape %0 : tensor<16x64x224x224xf32> -> tensor<16x64x50176xf32>
    %3 = flow.dispatch @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_0::@f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_0_generic_16x50176x64_f32(%2) : (tensor<16x64x50176xf32>) -> tensor<16x50176x64xf32>
    %4 = flow.tensor.reshape %3 : tensor<16x50176x64xf32> -> tensor<16x224x224x64xf32>
    %5 = flow.tensor.reshape %1 : tensor<128x64x7x7xf32> -> tensor<128x64x49xf32>
    %6 = flow.dispatch @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_1::@f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_1_generic_49x64x128_f32(%5) : (tensor<128x64x49xf32>) -> tensor<49x64x128xf32>
    %7 = flow.tensor.reshape %6 : tensor<49x64x128xf32> -> tensor<7x7x64x128xf32>
    %8 = flow.tensor.splat %cst : tensor<16x230x230x64xf32>
    %9 = flow.dispatch @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_2::@f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_2(%4, %8) : (tensor<16x224x224x64xf32>, tensor<16x230x230x64xf32>) -> %8
    %10 = flow.dispatch @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_3::@f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_16x112x112x128x7x7x64_f32(%9, %7) : (tensor<16x230x230x64xf32>, tensor<7x7x64x128xf32>) -> tensor<16x112x112x128xf32>
    %11 = flow.tensor.reshape %10 : tensor<16x112x112x128xf32> -> tensor<16x12544x128xf32>
    %12 = flow.dispatch @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_4::@f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_dispatch_4_generic_16x128x12544_f32(%11) : (tensor<16x12544x128xf32>) -> tensor<16x128x12544xf32>
    %13 = flow.tensor.reshape %12 : tensor<16x128x12544xf32> -> tensor<16x128x112x112xf32>
    %14 = hal.tensor.export %13 "output 0" : tensor<16x128x112x112xf32> -> !hal.buffer_view
    return %14 : !hal.buffer_view
  }
}
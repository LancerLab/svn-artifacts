module @f_17_mobilenet_128x96x112x112_112x112_112x112 {
  flow.executable private @f_17_mobilenet_128x96x112x112_112x112_112x112_dispatch_0 {
    flow.executable.export public @f_17_mobilenet_128x96x112x112_112x112_112x112_dispatch_0_generic_12288x12544_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_17_mobilenet_128x96x112x112_112x112_112x112_dispatch_0_generic_12288x12544_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<12288x12544xf32>>, %arg1: !flow.dispatch.tensor<readwrite:tensor<12288xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [12288, 12544], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<12288x12544xf32>> -> tensor<12288x12544xf32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0], sizes = [12288], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<12288xf32>> -> tensor<12288xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%0 : tensor<12288x12544xf32>) outs(%1 : tensor<12288xf32>) {
        ^bb0(%in: f32, %out: f32):
          %3 = arith.addf %out, %in : f32
          linalg.yield %3 : f32
        } -> tensor<12288xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [12288], strides = [1] : tensor<12288xf32> -> !flow.dispatch.tensor<readwrite:tensor<12288xf32>>
        return
      }
    }
  }
  flow.executable private @f_17_mobilenet_128x96x112x112_112x112_112x112_dispatch_1 {
    flow.executable.export public @f_17_mobilenet_128x96x112x112_112x112_112x112_dispatch_1_generic_154140672_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_17_mobilenet_128x96x112x112_112x112_112x112_dispatch_1_generic_154140672_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<154140672xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<154140672xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0], sizes = [154140672], strides = [1] : !flow.dispatch.tensor<readonly:tensor<154140672xf32>> -> tensor<154140672xf32>
        %1 = tensor.empty() : tensor<154140672xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0 : tensor<154140672xf32>) outs(%1 : tensor<154140672xf32>) {
        ^bb0(%in: f32, %out: f32):
          %3 = arith.mulf %in, %in : f32
          linalg.yield %3 : f32
        } -> tensor<154140672xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [154140672], strides = [1] : tensor<154140672xf32> -> !flow.dispatch.tensor<writeonly:tensor<154140672xf32>>
        return
      }
    }
  }
  flow.executable private @f_17_mobilenet_128x96x112x112_112x112_112x112_dispatch_3 {
    flow.executable.export public @f_17_mobilenet_128x96x112x112_112x112_112x112_dispatch_3_generic_128x96x112x112_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_17_mobilenet_128x96x112x112_112x112_112x112_dispatch_3_generic_128x96x112x112_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<128x96x112x112xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<128x96xf32>>, %arg2: !flow.dispatch.tensor<readonly:tensor<128x96xf32>>, %arg3: !flow.dispatch.tensor<readonly:tensor<112x112xf32>>, %arg4: !flow.dispatch.tensor<readonly:tensor<112x112xf32>>, %arg5: !flow.dispatch.tensor<writeonly:tensor<128x96x112x112xf32>>) {
        %cst = arith.constant 1.254400e+04 : f32
        %cst_0 = arith.constant 9.99999974E-6 : f32
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [128, 96, 112, 112], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x96x112x112xf32>> -> tensor<128x96x112x112xf32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [128, 96], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x96xf32>> -> tensor<128x96xf32>
        %2 = flow.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [128, 96], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x96xf32>> -> tensor<128x96xf32>
        %3 = flow.dispatch.tensor.load %arg3, offsets = [0, 0], sizes = [112, 112], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<112x112xf32>> -> tensor<112x112xf32>
        %4 = flow.dispatch.tensor.load %arg4, offsets = [0, 0], sizes = [112, 112], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<112x112xf32>> -> tensor<112x112xf32>
        %5 = tensor.empty() : tensor<128x96x112x112xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0, %1, %2, %3, %4 : tensor<128x96x112x112xf32>, tensor<128x96xf32>, tensor<128x96xf32>, tensor<112x112xf32>, tensor<112x112xf32>) outs(%5 : tensor<128x96x112x112xf32>) {
        ^bb0(%in: f32, %in_1: f32, %in_2: f32, %in_3: f32, %in_4: f32, %out: f32):
          %7 = arith.divf %in_1, %cst : f32
          %8 = arith.mulf %7, %7 : f32
          %9 = arith.divf %in_2, %cst : f32
          %10 = arith.subf %9, %8 : f32
          %11 = arith.addf %10, %cst_0 : f32
          %12 = math.rsqrt %11 : f32
          %13 = arith.subf %in, %7 : f32
          %14 = arith.mulf %13, %12 : f32
          %15 = arith.mulf %14, %in_3 : f32
          %16 = arith.addf %15, %in_4 : f32
          linalg.yield %16 : f32
        } -> tensor<128x96x112x112xf32>
        flow.dispatch.tensor.store %6, %arg5, offsets = [0, 0, 0, 0], sizes = [128, 96, 112, 112], strides = [1, 1, 1, 1] : tensor<128x96x112x112xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x96x112x112xf32>>
        return
      }
    }
  }
  func.func @f_17_mobilenet_128x96x112x112_112x112_112x112(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<128x96x112x112xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<112x112xf32>
    %2 = hal.tensor.import %arg2 "input 2" : !hal.buffer_view -> tensor<112x112xf32>
    %3 = flow.tensor.splat %cst : tensor<12288xf32>
    %4 = flow.tensor.reshape %0 : tensor<128x96x112x112xf32> -> tensor<12288x12544xf32>
    %5 = flow.dispatch @f_17_mobilenet_128x96x112x112_112x112_112x112_dispatch_0::@f_17_mobilenet_128x96x112x112_112x112_112x112_dispatch_0_generic_12288x12544_f32(%4, %3) : (tensor<12288x12544xf32>, tensor<12288xf32>) -> %3
    %6 = flow.tensor.reshape %5 : tensor<12288xf32> -> tensor<128x96xf32>
    %7 = flow.tensor.reshape %0 : tensor<128x96x112x112xf32> -> tensor<154140672xf32>
    %8 = flow.dispatch @f_17_mobilenet_128x96x112x112_112x112_112x112_dispatch_1::@f_17_mobilenet_128x96x112x112_112x112_112x112_dispatch_1_generic_154140672_f32(%7) : (tensor<154140672xf32>) -> tensor<154140672xf32>
    %9 = flow.tensor.reshape %8 : tensor<154140672xf32> -> tensor<12288x12544xf32>
    %10 = flow.dispatch @f_17_mobilenet_128x96x112x112_112x112_112x112_dispatch_0::@f_17_mobilenet_128x96x112x112_112x112_112x112_dispatch_0_generic_12288x12544_f32(%9, %3) : (tensor<12288x12544xf32>, tensor<12288xf32>) -> %3
    %11 = flow.tensor.reshape %10 : tensor<12288xf32> -> tensor<128x96xf32>
    %12 = flow.dispatch @f_17_mobilenet_128x96x112x112_112x112_112x112_dispatch_3::@f_17_mobilenet_128x96x112x112_112x112_112x112_dispatch_3_generic_128x96x112x112_f32(%0, %6, %11, %1, %2) : (tensor<128x96x112x112xf32>, tensor<128x96xf32>, tensor<128x96xf32>, tensor<112x112xf32>, tensor<112x112xf32>) -> tensor<128x96x112x112xf32>
    %13 = hal.tensor.export %12 "output 0" : tensor<128x96x112x112xf32> -> !hal.buffer_view
    return %13 : !hal.buffer_view
  }
}
module @f_15_gpt_16x1024x4096_4096_4096 {
  flow.executable private @f_15_gpt_16x1024x4096_4096_4096_dispatch_0 {
    flow.executable.export public @f_15_gpt_16x1024x4096_4096_4096_dispatch_0_generic_16384x4096_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_15_gpt_16x1024x4096_4096_4096_dispatch_0_generic_16384x4096_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<16384x4096xf32>>, %arg1: !flow.dispatch.tensor<readwrite:tensor<16384xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [16384, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16384x4096xf32>> -> tensor<16384x4096xf32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0], sizes = [16384], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<16384xf32>> -> tensor<16384xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%0 : tensor<16384x4096xf32>) outs(%1 : tensor<16384xf32>) {
        ^bb0(%in: f32, %out: f32):
          %3 = arith.mulf %in, %in : f32
          %4 = arith.addf %out, %3 : f32
          linalg.yield %4 : f32
        } -> tensor<16384xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [16384], strides = [1] : tensor<16384xf32> -> !flow.dispatch.tensor<readwrite:tensor<16384xf32>>
        return
      }
    }
  }
  flow.executable private @f_15_gpt_16x1024x4096_4096_4096_dispatch_1 {
    flow.executable.export public @f_15_gpt_16x1024x4096_4096_4096_dispatch_1_generic_16x1024x4096_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_15_gpt_16x1024x4096_4096_4096_dispatch_1_generic_16x1024x4096_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<16x1024x4096xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<16x1024xf32>>, %arg2: !flow.dispatch.tensor<readonly:tensor<4096xf32>>, %arg3: !flow.dispatch.tensor<readonly:tensor<4096xf32>>, %arg4: !flow.dispatch.tensor<writeonly:tensor<16x1024x4096xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 4.096000e+03 : f32
        %cst_1 = arith.constant 9.99999974E-6 : f32
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [16, 1024, 4096], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x1024x4096xf32>> -> tensor<16x1024x4096xf32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [16, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x1024xf32>> -> tensor<16x1024xf32>
        %2 = flow.dispatch.tensor.load %arg2, offsets = [0], sizes = [4096], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4096xf32>> -> tensor<4096xf32>
        %3 = flow.dispatch.tensor.load %arg3, offsets = [0], sizes = [4096], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4096xf32>> -> tensor<4096xf32>
        %4 = tensor.empty() : tensor<16x1024x4096xf32>
        %5 = tensor.empty() : tensor<16x1024xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<16x1024xf32>) -> tensor<16x1024xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0 : tensor<16x1024x4096xf32>) outs(%6 : tensor<16x1024xf32>) {
        ^bb0(%in: f32, %out: f32):
          %9 = arith.addf %out, %in : f32
          linalg.yield %9 : f32
        } -> tensor<16x1024xf32>
        %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %7, %1, %2, %3 : tensor<16x1024x4096xf32>, tensor<16x1024xf32>, tensor<16x1024xf32>, tensor<4096xf32>, tensor<4096xf32>) outs(%4 : tensor<16x1024x4096xf32>) {
        ^bb0(%in: f32, %in_2: f32, %in_3: f32, %in_4: f32, %in_5: f32, %out: f32):
          %9 = arith.divf %in_2, %cst_0 : f32
          %10 = arith.mulf %9, %9 : f32
          %11 = arith.divf %in_3, %cst_0 : f32
          %12 = arith.subf %11, %10 : f32
          %13 = arith.addf %12, %cst_1 : f32
          %14 = math.rsqrt %13 : f32
          %15 = arith.subf %in, %9 : f32
          %16 = arith.mulf %15, %14 : f32
          %17 = arith.mulf %16, %in_4 : f32
          %18 = arith.addf %17, %in_5 : f32
          linalg.yield %18 : f32
        } -> tensor<16x1024x4096xf32>
        flow.dispatch.tensor.store %8, %arg4, offsets = [0, 0, 0], sizes = [16, 1024, 4096], strides = [1, 1, 1] : tensor<16x1024x4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x1024x4096xf32>>
        return
      }
    }
  }
  func.func @f_15_gpt_16x1024x4096_4096_4096(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<16x1024x4096xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<4096xf32>
    %2 = hal.tensor.import %arg2 "input 2" : !hal.buffer_view -> tensor<4096xf32>
    %3 = flow.tensor.splat %cst : tensor<16384xf32>
    %4 = flow.tensor.reshape %0 : tensor<16x1024x4096xf32> -> tensor<16384x4096xf32>
    %5 = flow.dispatch @f_15_gpt_16x1024x4096_4096_4096_dispatch_0::@f_15_gpt_16x1024x4096_4096_4096_dispatch_0_generic_16384x4096_f32(%4, %3) : (tensor<16384x4096xf32>, tensor<16384xf32>) -> %3
    %6 = flow.tensor.reshape %5 : tensor<16384xf32> -> tensor<16x1024xf32>
    %7 = flow.dispatch @f_15_gpt_16x1024x4096_4096_4096_dispatch_1::@f_15_gpt_16x1024x4096_4096_4096_dispatch_1_generic_16x1024x4096_f32(%0, %6, %1, %2) : (tensor<16x1024x4096xf32>, tensor<16x1024xf32>, tensor<4096xf32>, tensor<4096xf32>) -> tensor<16x1024x4096xf32>
    %8 = hal.tensor.export %7 "output 0" : tensor<16x1024x4096xf32> -> !hal.buffer_view
    return %8 : !hal.buffer_view
  }
}
module @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D {
  flow.executable private @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_0 {
    flow.executable.export public @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_0_generic_Dx56x56x64_f32 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_0_generic_Dx56x56x64_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<?x64x56x56xf32>>, %arg1: index, %arg2: !flow.dispatch.tensor<writeonly:tensor<?x56x56x64xf32>>) {
        %0 = flow.dispatch.workload.ordinal %arg1, 0 : index
        %1 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x64x56x56xf32>>{%0}
        %2 = flow.dispatch.tie_shape %arg2 : !flow.dispatch.tensor<writeonly:tensor<?x56x56x64xf32>>{%0}
        %3 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [%0, 64, 56, 56], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x64x56x56xf32>>{%0} -> tensor<?x64x56x56xf32>
        %4 = tensor.empty(%0) : tensor<?x56x56x64xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<?x64x56x56xf32>) outs(%4 : tensor<?x56x56x64xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<?x56x56x64xf32>
        flow.dispatch.tensor.store %5, %2, offsets = [0, 0, 0, 0], sizes = [%0, 56, 56, 64], strides = [1, 1, 1, 1] : tensor<?x56x56x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x56x56x64xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_1 {
    flow.executable.export public @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_1_generic_9x64x128_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_1_generic_9x64x128_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<128x64x9xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<9x64x128xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [128, 64, 9], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x64x9xf32>> -> tensor<128x64x9xf32>
        %1 = tensor.empty() : tensor<9x64x128xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<128x64x9xf32>) outs(%1 : tensor<9x64x128xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<9x64x128xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [9, 64, 128], strides = [1, 1, 1] : tensor<9x64x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<9x64x128xf32>>
        return
      }
    }
  }
  flow.executable private @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_2 {
    flow.executable.export public @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_2 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_2(%arg0: !flow.dispatch.tensor<readonly:tensor<?x56x56x64xf32>>, %arg1: !flow.dispatch.tensor<readwrite:tensor<?x58x58x64xf32>>, %arg2: index) {
        %0 = flow.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x56x56x64xf32>>{%0}
        %2 = flow.dispatch.tie_shape %arg1 : !flow.dispatch.tensor<readwrite:tensor<?x58x58x64xf32>>{%0}
        %3 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [%0, 56, 56, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x56x56x64xf32>>{%0} -> tensor<?x56x56x64xf32>
        flow.dispatch.tensor.store %3, %2, offsets = [0, 1, 1, 0], sizes = [%0, 56, 56, 64], strides = [1, 1, 1, 1] : tensor<?x56x56x64xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x58x58x64xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_3 {
    flow.executable.export public @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_Dx56x56x128x3x3x64_f32 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_Dx56x56x128x3x3x64_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<?x58x58x64xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<3x3x64x128xf32>>, %arg2: index, %arg3: !flow.dispatch.tensor<writeonly:tensor<?x56x56x128xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = flow.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x58x58x64xf32>>{%0}
        %2 = flow.dispatch.tie_shape %arg3 : !flow.dispatch.tensor<writeonly:tensor<?x56x56x128xf32>>{%0}
        %3 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [%0, 58, 58, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x58x58x64xf32>>{%0} -> tensor<?x58x58x64xf32>
        %4 = flow.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [3, 3, 64, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x64x128xf32>> -> tensor<3x3x64x128xf32>
        %5 = tensor.empty(%0) : tensor<?x56x56x128xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<?x56x56x128xf32>) -> tensor<?x56x56x128xf32>
        %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<?x58x58x64xf32>, tensor<3x3x64x128xf32>) outs(%6 : tensor<?x56x56x128xf32>) -> tensor<?x56x56x128xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [%0, 56, 56, 128], strides = [1, 1, 1, 1] : tensor<?x56x56x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x56x56x128xf32>>{%0}
        return
      }
    }
  }
  flow.executable private @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_4 {
    flow.executable.export public @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_4_generic_Dx128x56x56_f32 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_4_generic_Dx128x56x56_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<?x56x56x128xf32>>, %arg1: index, %arg2: !flow.dispatch.tensor<writeonly:tensor<?x128x56x56xf32>>) {
        %0 = flow.dispatch.workload.ordinal %arg1, 0 : index
        %1 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x56x56x128xf32>>{%0}
        %2 = flow.dispatch.tie_shape %arg2 : !flow.dispatch.tensor<writeonly:tensor<?x128x56x56xf32>>{%0}
        %3 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [%0, 56, 56, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x56x56x128xf32>>{%0} -> tensor<?x56x56x128xf32>
        %4 = tensor.empty(%0) : tensor<?x128x56x56xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<?x56x56x128xf32>) outs(%4 : tensor<?x128x56x56xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<?x128x56x56xf32>
        flow.dispatch.tensor.store %5, %2, offsets = [0, 0, 0, 0], sizes = [%0, 128, 56, 56], strides = [1, 1, 1, 1] : tensor<?x128x56x56xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x128x56x56xf32>>{%0}
        return
      }
    }
  }
  func.func @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<?x64x56x56xf32>{%0}
    %2 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<128x64x3x3xf32>
    %3 = flow.dispatch @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_0::@f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_0_generic_Dx56x56x64_f32[%0](%1, %0) : (tensor<?x64x56x56xf32>{%0}, index) -> tensor<?x56x56x64xf32>{%0}
    %4 = flow.tensor.reshape %2 : tensor<128x64x3x3xf32> -> tensor<128x64x9xf32>
    %5 = flow.dispatch @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_1::@f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_1_generic_9x64x128_f32(%4) : (tensor<128x64x9xf32>) -> tensor<9x64x128xf32>
    %6 = flow.tensor.reshape %5 : tensor<9x64x128xf32> -> tensor<3x3x64x128xf32>
    %7 = flow.tensor.splat %cst : tensor<?x58x58x64xf32>{%0}
    %8 = flow.dispatch @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_2::@f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_2[%0](%3, %7, %0) : (tensor<?x56x56x64xf32>{%0}, tensor<?x58x58x64xf32>{%0}, index) -> %7{%0}
    %9 = flow.dispatch @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_3::@f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_3_conv_2d_nhwc_hwcf_Dx56x56x128x3x3x64_f32[%0](%8, %6, %0) : (tensor<?x58x58x64xf32>{%0}, tensor<3x3x64x128xf32>, index) -> tensor<?x56x56x128xf32>{%0}
    %10 = flow.dispatch @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_4::@f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D_dispatch_4_generic_Dx128x56x56_f32[%0](%9, %0) : (tensor<?x56x56x128xf32>{%0}, index) -> tensor<?x128x56x56xf32>{%0}
    %11 = hal.tensor.export %10 "output 0" : tensor<?x128x56x56xf32>{%0} -> !hal.buffer_view
    return %11 : !hal.buffer_view
  }
}
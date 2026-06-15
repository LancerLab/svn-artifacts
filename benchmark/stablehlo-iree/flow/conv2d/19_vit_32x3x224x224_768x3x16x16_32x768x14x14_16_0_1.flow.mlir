module @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1 {
  flow.executable private @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_0 {
    flow.executable.export public @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_0_generic_32x50176x3_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_0_generic_32x50176x3_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<32x3x50176xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<32x50176x3xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 3, 50176], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x3x50176xf32>> -> tensor<32x3x50176xf32>
        %1 = tensor.empty() : tensor<32x50176x3xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<32x3x50176xf32>) outs(%1 : tensor<32x50176x3xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<32x50176x3xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [32, 50176, 3], strides = [1, 1, 1] : tensor<32x50176x3xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x50176x3xf32>>
        return
      }
    }
  }
  flow.executable private @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_1 {
    flow.executable.export public @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_1_generic_256x3x768_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_1_generic_256x3x768_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<768x3x256xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<256x3x768xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [768, 3, 256], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<768x3x256xf32>> -> tensor<768x3x256xf32>
        %1 = tensor.empty() : tensor<256x3x768xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<768x3x256xf32>) outs(%1 : tensor<256x3x768xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<256x3x768xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [256, 3, 768], strides = [1, 1, 1] : tensor<256x3x768xf32> -> !flow.dispatch.tensor<writeonly:tensor<256x3x768xf32>>
        return
      }
    }
  }
  flow.executable private @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_2 {
    flow.executable.export public @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_2_conv_2d_nhwc_hwcf_32x14x14x768x16x16x3_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_2_conv_2d_nhwc_hwcf_32x14x14x768x16x16x3_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<32x224x224x3xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<16x16x3x768xf32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<32x14x14x768xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [32, 224, 224, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x224x224x3xf32>> -> tensor<32x224x224x3xf32>
        %1 = flow.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [16, 16, 3, 768], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x16x3x768xf32>> -> tensor<16x16x3x768xf32>
        %2 = tensor.empty() : tensor<32x14x14x768xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32x14x14x768xf32>) -> tensor<32x14x14x768xf32>
        %4 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<16> : tensor<2xi64>} ins(%0, %1 : tensor<32x224x224x3xf32>, tensor<16x16x3x768xf32>) outs(%3 : tensor<32x14x14x768xf32>) -> tensor<32x14x14x768xf32>
        flow.dispatch.tensor.store %4, %arg2, offsets = [0, 0, 0, 0], sizes = [32, 14, 14, 768], strides = [1, 1, 1, 1] : tensor<32x14x14x768xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x14x14x768xf32>>
        return
      }
    }
  }
  flow.executable private @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_3 {
    flow.executable.export public @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_3_generic_32x768x196_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_3_generic_32x768x196_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<32x196x768xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<32x768x196xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 196, 768], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x196x768xf32>> -> tensor<32x196x768xf32>
        %1 = tensor.empty() : tensor<32x768x196xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<32x196x768xf32>) outs(%1 : tensor<32x768x196xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<32x768x196xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [32, 768, 196], strides = [1, 1, 1] : tensor<32x768x196xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x768x196xf32>>
        return
      }
    }
  }
  func.func @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<32x3x224x224xf32>
    %1 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<768x3x16x16xf32>
    %2 = flow.tensor.reshape %0 : tensor<32x3x224x224xf32> -> tensor<32x3x50176xf32>
    %3 = flow.dispatch @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_0::@f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_0_generic_32x50176x3_f32(%2) : (tensor<32x3x50176xf32>) -> tensor<32x50176x3xf32>
    %4 = flow.tensor.reshape %3 : tensor<32x50176x3xf32> -> tensor<32x224x224x3xf32>
    %5 = flow.tensor.reshape %1 : tensor<768x3x16x16xf32> -> tensor<768x3x256xf32>
    %6 = flow.dispatch @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_1::@f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_1_generic_256x3x768_f32(%5) : (tensor<768x3x256xf32>) -> tensor<256x3x768xf32>
    %7 = flow.tensor.reshape %6 : tensor<256x3x768xf32> -> tensor<16x16x3x768xf32>
    %8 = flow.dispatch @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_2::@f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_2_conv_2d_nhwc_hwcf_32x14x14x768x16x16x3_f32(%4, %7) : (tensor<32x224x224x3xf32>, tensor<16x16x3x768xf32>) -> tensor<32x14x14x768xf32>
    %9 = flow.tensor.reshape %8 : tensor<32x14x14x768xf32> -> tensor<32x196x768xf32>
    %10 = flow.dispatch @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_3::@f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_dispatch_3_generic_32x768x196_f32(%9) : (tensor<32x196x768xf32>) -> tensor<32x768x196xf32>
    %11 = flow.tensor.reshape %10 : tensor<32x768x196xf32> -> tensor<32x768x14x14xf32>
    %12 = hal.tensor.export %11 "output 0" : tensor<32x768x14x14xf32> -> !hal.buffer_view
    return %12 : !hal.buffer_view
  }
}
module {
  func.func @f_18_unet_16x64x128x128_128x64x3x3_16x128x128x128_S_P_D(%input: tensor<16x64x128x128xf32>, %filter: tensor<128x64x3x3xf32>) -> tensor<16x128x128x128xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<16x64x128x128xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<16x64x128x128xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<16x64x128x128xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<16x64x128x128xf32>
    %filter_d0 = tensor.dim %filter, %c0 : tensor<128x64x3x3xf32>
    %filter_d1 = tensor.dim %filter, %c1 : tensor<128x64x3x3xf32>
    %filter_d2 = tensor.dim %filter, %c2 : tensor<128x64x3x3xf32>
    %filter_d3 = tensor.dim %filter, %c3 : tensor<128x64x3x3xf32>
    %out = tensor.empty() : tensor<16x128x128x128xf32>
    %pad_cst = arith.constant 0.0 : f32
    %padded = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%pi0: index, %pi1: index, %pi2: index, %pi3: index):
      tensor.yield %pad_cst : f32
    } : tensor<16x64x128x128xf32> to tensor<16x64x130x130xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%out : tensor<16x128x128x128xf32>) -> tensor<16x128x128x128xf32>
    %result = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%padded, %filter : tensor<16x64x130x130xf32>, tensor<128x64x3x3xf32>)
      outs(%filled : tensor<16x128x128x128xf32>) -> tensor<16x128x128x128xf32>
    return %result : tensor<16x128x128x128xf32>
  }
}

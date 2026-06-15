module {
  func.func @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D(%input: tensor<32x128x112x112xf32>, %filter: tensor<256x128x3x3xf32>) -> tensor<32x256x56x56xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<32x128x112x112xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<32x128x112x112xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<32x128x112x112xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<32x128x112x112xf32>
    %filter_d0 = tensor.dim %filter, %c0 : tensor<256x128x3x3xf32>
    %filter_d1 = tensor.dim %filter, %c1 : tensor<256x128x3x3xf32>
    %filter_d2 = tensor.dim %filter, %c2 : tensor<256x128x3x3xf32>
    %filter_d3 = tensor.dim %filter, %c3 : tensor<256x128x3x3xf32>
    %out = tensor.empty() : tensor<32x256x56x56xf32>
    %pad_cst = arith.constant 0.0 : f32
    %padded = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%pi0: index, %pi1: index, %pi2: index, %pi3: index):
      tensor.yield %pad_cst : f32
    } : tensor<32x128x112x112xf32> to tensor<32x128x114x114xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%out : tensor<32x256x56x56xf32>) -> tensor<32x256x56x56xf32>
    %result = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%padded, %filter : tensor<32x128x114x114xf32>, tensor<256x128x3x3xf32>)
      outs(%filled : tensor<32x256x56x56xf32>) -> tensor<32x256x56x56xf32>
    return %result : tensor<32x256x56x56xf32>
  }
}

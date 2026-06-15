module {
  func.func @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D(%input: tensor<?x64x56x56xf32>, %filter: tensor<128x64x3x3xf32>) -> tensor<?x128x56x56xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<?x64x56x56xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<?x64x56x56xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<?x64x56x56xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<?x64x56x56xf32>
    %filter_d0 = tensor.dim %filter, %c0 : tensor<128x64x3x3xf32>
    %filter_d1 = tensor.dim %filter, %c1 : tensor<128x64x3x3xf32>
    %filter_d2 = tensor.dim %filter, %c2 : tensor<128x64x3x3xf32>
    %filter_d3 = tensor.dim %filter, %c3 : tensor<128x64x3x3xf32>
    %out = tensor.empty(%input_d0) : tensor<?x128x56x56xf32>
    %pad_cst = arith.constant 0.0 : f32
    %padded = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%pi0: index, %pi1: index, %pi2: index, %pi3: index):
      tensor.yield %pad_cst : f32
    } : tensor<?x64x56x56xf32> to tensor<?x64x58x58xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%out : tensor<?x128x56x56xf32>) -> tensor<?x128x56x56xf32>
    %result = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%padded, %filter : tensor<?x64x58x58xf32>, tensor<128x64x3x3xf32>)
      outs(%filled : tensor<?x128x56x56xf32>) -> tensor<?x128x56x56xf32>
    return %result : tensor<?x128x56x56xf32>
  }
}

module {
  func.func @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1(%input: tensor<32x3x224x224xf32>, %filter: tensor<768x3x16x16xf32>) -> tensor<32x768x14x14xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<32x3x224x224xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<32x3x224x224xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<32x3x224x224xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<32x3x224x224xf32>
    %filter_d0 = tensor.dim %filter, %c0 : tensor<768x3x16x16xf32>
    %filter_d1 = tensor.dim %filter, %c1 : tensor<768x3x16x16xf32>
    %filter_d2 = tensor.dim %filter, %c2 : tensor<768x3x16x16xf32>
    %filter_d3 = tensor.dim %filter, %c3 : tensor<768x3x16x16xf32>
    %out = tensor.empty() : tensor<32x768x14x14xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%out : tensor<32x768x14x14xf32>) -> tensor<32x768x14x14xf32>
    %result = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<16> : vector<2xi64>}
      ins(%input, %filter : tensor<32x3x224x224xf32>, tensor<768x3x16x16xf32>)
      outs(%filled : tensor<32x768x14x14xf32>) -> tensor<32x768x14x14xf32>
    return %result : tensor<32x768x14x14xf32>
  }
}

module {
  func.func @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1(%input: tensor<8x256x64x64xf32>, %filter: tensor<21x256x1x1xf32>) -> tensor<8x21x64x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<8x256x64x64xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<8x256x64x64xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<8x256x64x64xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<8x256x64x64xf32>
    %filter_d0 = tensor.dim %filter, %c0 : tensor<21x256x1x1xf32>
    %filter_d1 = tensor.dim %filter, %c1 : tensor<21x256x1x1xf32>
    %filter_d2 = tensor.dim %filter, %c2 : tensor<21x256x1x1xf32>
    %filter_d3 = tensor.dim %filter, %c3 : tensor<21x256x1x1xf32>
    %out = tensor.empty() : tensor<8x21x64x64xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%out : tensor<8x21x64x64xf32>) -> tensor<8x21x64x64xf32>
    %result = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<8x256x64x64xf32>, tensor<21x256x1x1xf32>)
      outs(%filled : tensor<8x21x64x64xf32>) -> tensor<8x21x64x64xf32>
    return %result : tensor<8x21x64x64xf32>
  }
}

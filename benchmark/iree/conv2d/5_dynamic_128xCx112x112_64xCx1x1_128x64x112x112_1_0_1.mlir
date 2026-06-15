module {
  func.func @f_5_dynamic_128xCx112x112_64xCx1x1_128x64x112x112_1_0_1(%input: tensor<128x?x112x112xf32>, %filter: tensor<64x?x1x1xf32>) -> tensor<128x64x112x112xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<128x?x112x112xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<128x?x112x112xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<128x?x112x112xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<128x?x112x112xf32>
    %filter_d0 = tensor.dim %filter, %c0 : tensor<64x?x1x1xf32>
    %filter_d1 = tensor.dim %filter, %c1 : tensor<64x?x1x1xf32>
    %filter_d2 = tensor.dim %filter, %c2 : tensor<64x?x1x1xf32>
    %filter_d3 = tensor.dim %filter, %c3 : tensor<64x?x1x1xf32>
    %out = tensor.empty() : tensor<128x64x112x112xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32) outs(%out : tensor<128x64x112x112xf32>) -> tensor<128x64x112x112xf32>
    %result = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<128x?x112x112xf32>, tensor<64x?x1x1xf32>)
      outs(%filled : tensor<128x64x112x112xf32>) -> tensor<128x64x112x112xf32>
    return %result : tensor<128x64x112x112xf32>
  }
}

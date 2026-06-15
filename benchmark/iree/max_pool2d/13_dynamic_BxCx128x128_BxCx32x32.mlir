module {
  func.func @f_13_dynamic_BxCx128x128_BxCx32x32(%input: tensor<?x?x128x128xf32>) -> tensor<?x?x32x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<?x?x128x128xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<?x?x128x128xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<?x?x128x128xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<?x?x128x128xf32>
    %out = tensor.empty(%input_d0, %input_d1) : tensor<?x?x32x32xf32>
    %neg_inf = arith.constant -3.4028234663852886e+38 : f32
    %filled = linalg.fill ins(%neg_inf : f32) outs(%out : tensor<?x?x32x32xf32>) -> tensor<?x?x32x32xf32>
    %kernel = tensor.empty() : tensor<4x4xf32>
    %result = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<4> : vector<2xi64>}
      ins(%input, %kernel : tensor<?x?x128x128xf32>, tensor<4x4xf32>)
      outs(%filled : tensor<?x?x32x32xf32>) -> tensor<?x?x32x32xf32>
    return %result : tensor<?x?x32x32xf32>
  }
}

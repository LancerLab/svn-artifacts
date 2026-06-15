module {
  func.func @f_6_dynamic_Bx1280xHxW_Bx1280xHd2xWd2(%input: tensor<?x1280x?x?xf32>) -> tensor<?x1280x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<?x1280x?x?xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<?x1280x?x?xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<?x1280x?x?xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<?x1280x?x?xf32>
    %out = tensor.empty(%input_d0, %input_d2, %input_d3) : tensor<?x1280x?x?xf32>
    %neg_inf = arith.constant -3.4028234663852886e+38 : f32
    %filled = linalg.fill ins(%neg_inf : f32) outs(%out : tensor<?x1280x?x?xf32>) -> tensor<?x1280x?x?xf32>
    %kernel = tensor.empty() : tensor<2x2xf32>
    %result = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
      ins(%input, %kernel : tensor<?x1280x?x?xf32>, tensor<2x2xf32>)
      outs(%filled : tensor<?x1280x?x?xf32>) -> tensor<?x1280x?x?xf32>
    return %result : tensor<?x1280x?x?xf32>
  }
}

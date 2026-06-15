module {
  func.func @f_13_dynamic_BxCx128x128_BxCx32x32_correct(%input: tensor<?x?x128x128xf32>) -> tensor<?x?x32x32xf32> {
    %neg_inf = arith.constant -3.40282e+38 : f32
    %ci0 = arith.constant 0 : index
    %od0 = tensor.dim %input, %ci0 : tensor<?x?x128x128xf32>
    %ci1 = arith.constant 1 : index
    %od1 = tensor.dim %input, %ci1 : tensor<?x?x128x128xf32>
    %init = tensor.empty(%od0, %od1) : tensor<?x?x32x32xf32>
    %fill = linalg.fill ins(%neg_inf : f32) outs(%init : tensor<?x?x32x32xf32>) -> tensor<?x?x32x32xf32>
    %kernel = tensor.empty() : tensor<97x97xf32>
    %r = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%input, %kernel : tensor<?x?x128x128xf32>, tensor<97x97xf32>)
         outs(%fill : tensor<?x?x32x32xf32>) -> tensor<?x?x32x32xf32>
    return %r : tensor<?x?x32x32xf32>
  }
}

module {
  func.func @f_15_inception_64x192x28x28_64x192x14x14_correct(%input: tensor<64x192x28x28xf32>) -> tensor<64x192x14x14xf32> {
    %neg_inf = arith.constant -3.40282e+38 : f32
    %init = tensor.empty() : tensor<64x192x14x14xf32>
    %fill = linalg.fill ins(%neg_inf : f32) outs(%init : tensor<64x192x14x14xf32>) -> tensor<64x192x14x14xf32>
    %kernel = tensor.empty() : tensor<15x15xf32>
    %r = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%input, %kernel : tensor<64x192x28x28xf32>, tensor<15x15xf32>)
         outs(%fill : tensor<64x192x14x14xf32>) -> tensor<64x192x14x14xf32>
    return %r : tensor<64x192x14x14xf32>
  }
}

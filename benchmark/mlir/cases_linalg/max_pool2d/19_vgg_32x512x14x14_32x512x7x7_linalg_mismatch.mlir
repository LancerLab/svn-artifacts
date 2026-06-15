module {
  func.func @f_19_vgg_32x512x14x14_32x512x7x7_mismatch(%input: tensor<32x512x14x14xf32>) -> tensor<32x511x7x7xf32> {
    %neg_inf = arith.constant -3.40282e+38 : f32
    %init = tensor.empty() : tensor<32x511x7x7xf32>
    %fill = linalg.fill ins(%neg_inf : f32) outs(%init : tensor<32x511x7x7xf32>) -> tensor<32x511x7x7xf32>
    %kernel = tensor.empty() : tensor<8x8xf32>
    %r = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%input, %kernel : tensor<32x512x14x14xf32>, tensor<8x8xf32>)
         outs(%fill : tensor<32x511x7x7xf32>) -> tensor<32x511x7x7xf32>
    return %r : tensor<32x511x7x7xf32>
  }
}

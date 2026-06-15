module {
  func.func @f_2_cnn_128x128x28x28_128x128x14x14_mismatch(%input: tensor<128x128x28x28xf32>) -> tensor<128x127x14x14xf32> {
    %neg_inf = arith.constant -3.40282e+38 : f32
    %init = tensor.empty() : tensor<128x127x14x14xf32>
    %fill = linalg.fill ins(%neg_inf : f32) outs(%init : tensor<128x127x14x14xf32>) -> tensor<128x127x14x14xf32>
    %kernel = tensor.empty() : tensor<15x15xf32>
    %r = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%input, %kernel : tensor<128x128x28x28xf32>, tensor<15x15xf32>)
         outs(%fill : tensor<128x127x14x14xf32>) -> tensor<128x127x14x14xf32>
    return %r : tensor<128x127x14x14xf32>
  }
}

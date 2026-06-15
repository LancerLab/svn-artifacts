module {
  func.func @f_16_mobilenet_128x96x224x224_128x96x112x112_correct(%input: tensor<128x96x224x224xf32>) -> tensor<128x96x112x112xf32> {
    %neg_inf = arith.constant -3.40282e+38 : f32
    %init = tensor.empty() : tensor<128x96x112x112xf32>
    %fill = linalg.fill ins(%neg_inf : f32) outs(%init : tensor<128x96x112x112xf32>) -> tensor<128x96x112x112xf32>
    %kernel = tensor.empty() : tensor<113x113xf32>
    %r = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%input, %kernel : tensor<128x96x224x224xf32>, tensor<113x113xf32>)
         outs(%fill : tensor<128x96x112x112xf32>) -> tensor<128x96x112x112xf32>
    return %r : tensor<128x96x112x112xf32>
  }
}

module {
  func.func @f_7_dynamic_128xCx112x112_128xCx56x56_correct(%input: tensor<128x?x112x112xf32>) -> tensor<128x?x56x56xf32> {
    %neg_inf = arith.constant -3.40282e+38 : f32
    %ci1 = arith.constant 1 : index
    %od1 = tensor.dim %input, %ci1 : tensor<128x?x112x112xf32>
    %init = tensor.empty(%od1) : tensor<128x?x56x56xf32>
    %fill = linalg.fill ins(%neg_inf : f32) outs(%init : tensor<128x?x56x56xf32>) -> tensor<128x?x56x56xf32>
    %kernel = tensor.empty() : tensor<57x57xf32>
    %r = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%input, %kernel : tensor<128x?x112x112xf32>, tensor<57x57xf32>)
         outs(%fill : tensor<128x?x56x56xf32>) -> tensor<128x?x56x56xf32>
    return %r : tensor<128x?x56x56xf32>
  }
}

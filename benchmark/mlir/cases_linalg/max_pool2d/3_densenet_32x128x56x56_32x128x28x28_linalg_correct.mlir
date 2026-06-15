module {
  func.func @f_3_densenet_32x128x56x56_32x128x28x28_correct(%input: tensor<32x128x56x56xf32>) -> tensor<32x128x28x28xf32> {
    %neg_inf = arith.constant -3.40282e+38 : f32
    %init = tensor.empty() : tensor<32x128x28x28xf32>
    %fill = linalg.fill ins(%neg_inf : f32) outs(%init : tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf32>
    %kernel = tensor.empty() : tensor<29x29xf32>
    %r = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%input, %kernel : tensor<32x128x56x56xf32>, tensor<29x29xf32>)
         outs(%fill : tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xf32>
    return %r : tensor<32x128x28x28xf32>
  }
}

module {
  func.func @f_5_dynamic_Bx256x56x56_Bx256x28x28_correct(%input: tensor<?x256x56x56xf32>) -> tensor<?x256x28x28xf32> {
    %neg_inf = arith.constant -3.40282e+38 : f32
    %ci0 = arith.constant 0 : index
    %od0 = tensor.dim %input, %ci0 : tensor<?x256x56x56xf32>
    %init = tensor.empty(%od0) : tensor<?x256x28x28xf32>
    %fill = linalg.fill ins(%neg_inf : f32) outs(%init : tensor<?x256x28x28xf32>) -> tensor<?x256x28x28xf32>
    %kernel = tensor.empty() : tensor<29x29xf32>
    %r = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%input, %kernel : tensor<?x256x56x56xf32>, tensor<29x29xf32>)
         outs(%fill : tensor<?x256x28x28xf32>) -> tensor<?x256x28x28xf32>
    return %r : tensor<?x256x28x28xf32>
  }
}

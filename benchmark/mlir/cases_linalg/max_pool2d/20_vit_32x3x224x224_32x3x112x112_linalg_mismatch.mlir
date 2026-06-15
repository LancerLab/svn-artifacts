module {
  func.func @f_20_vit_32x3x224x224_32x3x112x112_mismatch(%input: tensor<32x3x224x224xf32>) -> tensor<32x2x112x112xf32> {
    %neg_inf = arith.constant -3.40282e+38 : f32
    %init = tensor.empty() : tensor<32x2x112x112xf32>
    %fill = linalg.fill ins(%neg_inf : f32) outs(%init : tensor<32x2x112x112xf32>) -> tensor<32x2x112x112xf32>
    %kernel = tensor.empty() : tensor<113x113xf32>
    %r = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%input, %kernel : tensor<32x3x224x224xf32>, tensor<113x113xf32>)
         outs(%fill : tensor<32x2x112x112xf32>) -> tensor<32x2x112x112xf32>
    return %r : tensor<32x2x112x112xf32>
  }
}

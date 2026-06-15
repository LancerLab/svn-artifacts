module {
  func.func @f_10_dynamic_32xCxHxW_32xCxHd5xWd5_correct(%input: tensor<32x?x?x?xf32>) -> tensor<32x?x?x?xf32> {
    %neg_inf = arith.constant -3.40282e+38 : f32
    %ci1 = arith.constant 1 : index
    %od1 = tensor.dim %input, %ci1 : tensor<32x?x?x?xf32>
    %ci2 = arith.constant 2 : index
    %id2 = tensor.dim %input, %ci2 : tensor<32x?x?x?xf32>
    %ks2 = arith.constant 2 : index
    %od2 = arith.subi %id2, %ks2 : index
    %ci3 = arith.constant 3 : index
    %id3 = tensor.dim %input, %ci3 : tensor<32x?x?x?xf32>
    %ks3 = arith.constant 2 : index
    %od3 = arith.subi %id3, %ks3 : index
    %init = tensor.empty(%od1, %od2, %od3) : tensor<32x?x?x?xf32>
    %fill = linalg.fill ins(%neg_inf : f32) outs(%init : tensor<32x?x?x?xf32>) -> tensor<32x?x?x?xf32>
    %kernel = tensor.empty() : tensor<3x3xf32>
    %r = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%input, %kernel : tensor<32x?x?x?xf32>, tensor<3x3xf32>)
         outs(%fill : tensor<32x?x?x?xf32>) -> tensor<32x?x?x?xf32>
    return %r : tensor<32x?x?x?xf32>
  }
}

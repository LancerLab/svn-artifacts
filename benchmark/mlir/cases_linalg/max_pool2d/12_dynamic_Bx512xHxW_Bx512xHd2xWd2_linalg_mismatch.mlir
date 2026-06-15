module {
  func.func @f_12_dynamic_Bx512xHxW_Bx512xHd2xWd2_mismatch(%input: tensor<?x512x?x?xf32>) -> tensor<?x511x?x?xf32> {
    %neg_inf = arith.constant -3.40282e+38 : f32
    %ci0 = arith.constant 0 : index
    %od0 = tensor.dim %input, %ci0 : tensor<?x512x?x?xf32>
    %ci2 = arith.constant 2 : index
    %id2 = tensor.dim %input, %ci2 : tensor<?x512x?x?xf32>
    %ks2 = arith.constant 2 : index
    %od2 = arith.subi %id2, %ks2 : index
    %ci3 = arith.constant 3 : index
    %id3 = tensor.dim %input, %ci3 : tensor<?x512x?x?xf32>
    %ks3 = arith.constant 2 : index
    %od3 = arith.subi %id3, %ks3 : index
    %init = tensor.empty(%od0, %od2, %od3) : tensor<?x511x?x?xf32>
    %fill = linalg.fill ins(%neg_inf : f32) outs(%init : tensor<?x511x?x?xf32>) -> tensor<?x511x?x?xf32>
    %kernel = tensor.empty() : tensor<3x3xf32>
    %r = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%input, %kernel : tensor<?x512x?x?xf32>, tensor<3x3xf32>)
         outs(%fill : tensor<?x511x?x?xf32>) -> tensor<?x511x?x?xf32>
    return %r : tensor<?x511x?x?xf32>
  }
}

module {
  func.func @f_2_general_128x50_10000x128_128x50x128_correct(%indices: tensor<128x50xi64>, %weights: tensor<10000x128xf32>) -> tensor<128x50x128xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = tensor.empty() : tensor<128x50x128xf32>
    %cst = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x50x128xf32>) -> tensor<128x50x128xf32>
    return %fill : tensor<128x50x128xf32>
  }
}

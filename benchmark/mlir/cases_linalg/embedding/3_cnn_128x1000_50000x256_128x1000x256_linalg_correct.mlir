module {
  func.func @f_3_cnn_128x1000_50000x256_128x1000x256_correct(%indices: tensor<128x1000xi64>, %weights: tensor<50000x256xf32>) -> tensor<128x1000x256xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = tensor.empty() : tensor<128x1000x256xf32>
    %cst = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x1000x256xf32>) -> tensor<128x1000x256xf32>
    return %fill : tensor<128x1000x256xf32>
  }
}

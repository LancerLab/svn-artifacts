module {
  func.func @f_20_general_64x100_10000x300_64x100x300_correct(%indices: tensor<64x100xi64>, %weights: tensor<10000x300xf32>) -> tensor<64x100x300xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = tensor.empty() : tensor<64x100x300xf32>
    %cst = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<64x100x300xf32>) -> tensor<64x100x300xf32>
    return %fill : tensor<64x100x300xf32>
  }
}

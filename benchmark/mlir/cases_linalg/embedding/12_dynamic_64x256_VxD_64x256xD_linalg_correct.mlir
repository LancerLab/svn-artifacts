module {
  func.func @f_12_dynamic_64x256_VxD_64x256xD_correct(%indices: tensor<64x256xi64>, %weights: tensor<?x?xf32>) -> tensor<64x256x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cw1 = arith.constant 1 : index
    %od2 = tensor.dim %weights, %cw1 : tensor<?x?xf32>
    %init = tensor.empty(%od2) : tensor<64x256x?xf32>
    %cst = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<64x256x?xf32>) -> tensor<64x256x?xf32>
    return %fill : tensor<64x256x?xf32>
  }
}

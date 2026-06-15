module {
  func.func @f_11_dynamic_64xN_40000x512_64xNx512_correct(%indices: tensor<64x?xi64>, %weights: tensor<40000x512xf32>) -> tensor<64x?x512xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ci1 = arith.constant 1 : index
    %od1 = tensor.dim %indices, %ci1 : tensor<64x?xi64>
    %init = tensor.empty(%od1) : tensor<64x?x512xf32>
    %cst = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<64x?x512xf32>) -> tensor<64x?x512xf32>
    return %fill : tensor<64x?x512xf32>
  }
}

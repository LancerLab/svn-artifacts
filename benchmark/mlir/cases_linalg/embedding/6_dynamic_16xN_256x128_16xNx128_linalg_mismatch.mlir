module {
  func.func @f_6_dynamic_16xN_256x128_16xNx128_mismatch(%indices: tensor<16x?xi64>, %weights: tensor<256x127xf32>) -> tensor<16x?x128xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ci1 = arith.constant 1 : index
    %od1 = tensor.dim %indices, %ci1 : tensor<16x?xi64>
    %init = tensor.empty(%od1) : tensor<16x?x128xf32>
    %cst = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<16x?x128xf32>) -> tensor<16x?x128xf32>
    return %fill : tensor<16x?x128xf32>
  }
}

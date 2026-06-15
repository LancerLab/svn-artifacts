module {
  func.func @f_9_dynamic_32xN_32000x768_32xNx768_mismatch(%indices: tensor<32x?xi64>, %weights: tensor<32000x767xf32>) -> tensor<32x?x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ci1 = arith.constant 1 : index
    %od1 = tensor.dim %indices, %ci1 : tensor<32x?xi64>
    %init = tensor.empty(%od1) : tensor<32x?x768xf32>
    %cst = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<32x?x768xf32>) -> tensor<32x?x768xf32>
    return %fill : tensor<32x?x768xf32>
  }
}

module {
  func.func @f_18_transformer_32x512_32000x512_32x512x512_mismatch(%indices: tensor<32x512xi64>, %weights: tensor<32000x511xf32>) -> tensor<32x512x512xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = tensor.empty() : tensor<32x512x512xf32>
    %cst = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<32x512x512xf32>) -> tensor<32x512x512xf32>
    return %fill : tensor<32x512x512xf32>
  }
}

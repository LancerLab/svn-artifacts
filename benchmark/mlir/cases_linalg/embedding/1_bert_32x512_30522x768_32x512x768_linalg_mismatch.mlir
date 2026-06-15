module {
  func.func @f_1_bert_32x512_30522x768_32x512x768_mismatch(%indices: tensor<32x512xi64>, %weights: tensor<30522x767xf32>) -> tensor<32x512x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = tensor.empty() : tensor<32x512x768xf32>
    %cst = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<32x512x768xf32>) -> tensor<32x512x768xf32>
    return %fill : tensor<32x512x768xf32>
  }
}

module {
  func.func @f_19_vit_32x197_1000x768_32x197x768_mismatch(%indices: tensor<32x197xi64>, %weights: tensor<1000x767xf32>) -> tensor<32x197x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = tensor.empty() : tensor<32x197x768xf32>
    %cst = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<32x197x768xf32>) -> tensor<32x197x768xf32>
    return %fill : tensor<32x197x768xf32>
  }
}

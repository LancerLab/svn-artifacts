module {
  func.func @f_17_general_16x512_25000x768_16x512x768_correct(%indices: tensor<16x512xi64>, %weights: tensor<25000x768xf32>) -> tensor<16x512x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = tensor.empty() : tensor<16x512x768xf32>
    %cst = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<16x512x768xf32>) -> tensor<16x512x768xf32>
    return %fill : tensor<16x512x768xf32>
  }
}

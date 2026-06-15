module {
  func.func @f_14_gpt_16x1024_50257x1024_16x1024x1024_correct(%indices: tensor<16x1024xi64>, %weights: tensor<50257x1024xf32>) -> tensor<16x1024x1024xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = tensor.empty() : tensor<16x1024x1024xf32>
    %cst = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<16x1024x1024xf32>) -> tensor<16x1024x1024xf32>
    return %fill : tensor<16x1024x1024xf32>
  }
}

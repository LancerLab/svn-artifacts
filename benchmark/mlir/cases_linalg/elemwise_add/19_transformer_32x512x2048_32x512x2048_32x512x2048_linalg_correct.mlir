module {
  func.func @f_19_transformer_32x512x2048_32x512x2048_32x512x2048_correct(%a: tensor<32x512x2048xf32>, %b: tensor<32x512x2048xf32>) -> tensor<32x512x2048xf32> {
    %init = tensor.empty() : tensor<32x512x2048xf32>
    %r = linalg.add ins(%a, %b : tensor<32x512x2048xf32>, tensor<32x512x2048xf32>)
                    outs(%init : tensor<32x512x2048xf32>) -> tensor<32x512x2048xf32>
    return %r : tensor<32x512x2048xf32>
  }
}

module {
  func.func @f_15_gpt_16x1024x4096_16x1024x4096_16x1024x4096_correct(%a: tensor<16x1024x4096xf32>, %b: tensor<16x1024x4096xf32>) -> tensor<16x1024x4096xf32> {
    %init = tensor.empty() : tensor<16x1024x4096xf32>
    %r = linalg.add ins(%a, %b : tensor<16x1024x4096xf32>, tensor<16x1024x4096xf32>)
                    outs(%init : tensor<16x1024x4096xf32>) -> tensor<16x1024x4096xf32>
    return %r : tensor<16x1024x4096xf32>
  }
}

module {
  func.func @f_15_gpt_16x1024x1024_16x16x1024x64(%input: tensor<16x1024x1024xf32>) -> tensor<16x16x1024x64xf32> {
    %flat = tensor.collapse_shape %input [[0, 1, 2]] : tensor<16x1024x1024xf32> into tensor<16777216xf32>
    %out = tensor.expand_shape %flat [[0, 1, 2, 3]] output_shape [16, 16, 1024, 64] : tensor<16777216xf32> into tensor<16x16x1024x64xf32>
    return %out : tensor<16x16x1024x64xf32>
  }
}

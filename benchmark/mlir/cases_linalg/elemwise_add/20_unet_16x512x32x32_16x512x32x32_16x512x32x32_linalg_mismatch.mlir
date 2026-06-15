module {
  func.func @f_20_unet_16x512x32x32_16x512x32x32_16x512x32x32_mismatch(%a: tensor<16x512x32x32xf32>, %b: tensor<15x512x32x32xf32>) -> tensor<16x512x32x32xf32> {
    %init = tensor.empty() : tensor<16x512x32x32xf32>
    %r = linalg.add ins(%a, %b : tensor<16x512x32x32xf32>, tensor<15x512x32x32xf32>)
                    outs(%init : tensor<16x512x32x32xf32>) -> tensor<16x512x32x32xf32>
    return %r : tensor<16x512x32x32xf32>
  }
}

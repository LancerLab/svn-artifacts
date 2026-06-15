module {
  func.func @f_20_unet_16x512x32x32_16x512x32x32_16x1024x32x32_mismatch(%in0: tensor<16x512x32x32xf32>, %in1: tensor<15x512x32x32xf32>) -> tensor<16x1024x32x32xf32> {
    %r = tensor.concat dim(1) %in0, %in1 : (tensor<16x512x32x32xf32>, tensor<15x512x32x32xf32>) -> tensor<16x1024x32x32xf32>
    return %r : tensor<16x1024x32x32xf32>
  }
}

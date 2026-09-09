module {
  func.func @f_20_unet_16x512x32x32_16x512x32x32_16x1024x32x32(%in0: tensor<16x512x32x32xf32>, %in1: tensor<16x512x32x32xf32>) -> tensor<16x1024x32x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %out = tensor.empty() : tensor<16x1024x32x32xf32>
    %ins0 = tensor.insert_slice %in0 into %out[%c0, %c0, %c0, %c0][16, 512, 32, 32][1, 1, 1, 1] : tensor<16x512x32x32xf32> into tensor<16x1024x32x32xf32>
    %coff512 = arith.constant 512 : index
    %ins1 = tensor.insert_slice %in1 into %ins0[%c0, %coff512, %c0, %c0][16, 512, 32, 32][1, 1, 1, 1] : tensor<16x512x32x32xf32> into tensor<16x1024x32x32xf32>
    return %ins1 : tensor<16x1024x32x32xf32>
  }
}

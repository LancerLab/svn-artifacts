module @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256 {
  func.func @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256(%in0: tensor<32x512x64xf32>, %in1: tensor<32x512x64xf32>, %in2: tensor<32x512x64xf32>, %in3: tensor<32x512x64xf32>) -> tensor<32x512x256xf32> {
    %result = stablehlo.concatenate %in0, %in1, %in2, %in3, dim = 2 : (tensor<32x512x64xf32>, tensor<32x512x64xf32>, tensor<32x512x64xf32>, tensor<32x512x64xf32>) -> tensor<32x512x256xf32>
    return %result : tensor<32x512x256xf32>
  }
}

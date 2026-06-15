module @f_12_dynamic_64xSx256_Sx64x256 {
  func.func @f_12_dynamic_64xSx256_Sx64x256(%input: tensor<64x?x256xf32>) -> tensor<?x64x256xf32> {
    %result = stablehlo.transpose %input, dims = [1, 0, 2] : (tensor<64x?x256xf32>) -> tensor<?x64x256xf32>
    return %result : tensor<?x64x256xf32>
  }
}

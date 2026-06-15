module @f_21_vit_32x3x224x224_32x224x224x3 {
  func.func @f_21_vit_32x3x224x224_32x224x224x3(%input: tensor<32x3x224x224xf32>) -> tensor<32x224x224x3xf32> {
    %result = stablehlo.transpose %input, dims = [0, 2, 3, 1] : (tensor<32x3x224x224xf32>) -> tensor<32x224x224x3xf32>
    return %result : tensor<32x224x224x3xf32>
  }
}

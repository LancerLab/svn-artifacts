module @f_21_vit_32x3x224x224_32x196x768 {
  func.func @f_21_vit_32x3x224x224_32x196x768(%input: tensor<32x3x224x224xf32>) -> tensor<32x196x768xf32> {
    %result = stablehlo.reshape %input : (tensor<32x3x224x224xf32>) -> tensor<32x196x768xf32>
    return %result : tensor<32x196x768xf32>
  }
}

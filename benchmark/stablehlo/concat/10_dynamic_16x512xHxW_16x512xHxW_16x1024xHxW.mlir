module @f_10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW {
  func.func @f_10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW(%in0: tensor<16x512x?x?xf32>, %in1: tensor<16x512x?x?xf32>) -> tensor<16x1024x?x?xf32> {
    %result = stablehlo.concatenate %in0, %in1, dim = 1 : (tensor<16x512x?x?xf32>, tensor<16x512x?x?xf32>) -> tensor<16x1024x?x?xf32>
    return %result : tensor<16x1024x?x?xf32>
  }
}

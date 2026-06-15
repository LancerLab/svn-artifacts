module @f_18_resnet_128x2048_2048x1000_128x1000 {
  func.func @f_18_resnet_128x2048_2048x1000_128x1000(%input0: tensor<128x2048xf32>, %input1: tensor<2048x1000xf32>) -> tensor<128x1000xf32> {
    %result = stablehlo.dot %input0, %input1 : (tensor<128x2048xf32>, tensor<2048x1000xf32>) -> tensor<128x1000xf32>
    return %result : tensor<128x1000xf32>
  }
}

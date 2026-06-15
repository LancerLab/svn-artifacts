module @f_17_mobilenet_128x96x7x7_128x4704 {
  func.func @f_17_mobilenet_128x96x7x7_128x4704(%input: tensor<128x96x7x7xf32>) -> tensor<128x4704xf32> {
    %result = stablehlo.reshape %input : (tensor<128x96x7x7xf32>) -> tensor<128x4704xf32>
    return %result : tensor<128x4704xf32>
  }
}

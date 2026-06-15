module @f_5_dynamic_Bx2048_2048x1000_Bx1000 {
  func.func @f_5_dynamic_Bx2048_2048x1000_Bx1000(%input0: tensor<?x2048xf32>, %input1: tensor<2048x1000xf32>) -> tensor<?x1000xf32> {
    %result = stablehlo.dot %input0, %input1 : (tensor<?x2048xf32>, tensor<2048x1000xf32>) -> tensor<?x1000xf32>
    return %result : tensor<?x1000xf32>
  }
}

module @f_3_densenet_32x128x56x56_32x128x28x28 {
  func.func @f_3_densenet_32x128x56x56_32x128x28x28(%input: tensor<32x128x56x56xf32>) -> tensor<32x128x28x28xf32> {
    %neg_inf = stablehlo.constant dense<-3.402820e+38> : tensor<f32>
    %result = "stablehlo.reduce_window"(%input, %neg_inf) ({
      ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
        %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
        "stablehlo.return"(%m) : (tensor<f32>) -> ()
    }) {window_dimensions = dense<[1, 1, 2, 2]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>,
       padding = dense<0> : tensor<4x2xi64>, base_dilations = dense<1> : tensor<4xi64>, window_dilations = dense<1> : tensor<4xi64>}
        : (tensor<32x128x56x56xf32>, tensor<f32>) -> tensor<32x128x28x28xf32>
    return %result : tensor<32x128x28x28xf32>
  }
}

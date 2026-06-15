module @f_1_alexnet_64x256x27x27_64x256x13x13 {
  func.func @f_1_alexnet_64x256x27x27_64x256x13x13(%input: tensor<64x256x27x27xf32>) -> tensor<64x256x13x13xf32> {
    %neg_inf = stablehlo.constant dense<-3.402820e+38> : tensor<f32>
    %result = "stablehlo.reduce_window"(%input, %neg_inf) ({
      ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
        %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
        "stablehlo.return"(%m) : (tensor<f32>) -> ()
    }) {window_dimensions = dense<[1, 1, 2, 2]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>,
       padding = dense<0> : tensor<4x2xi64>, base_dilations = dense<1> : tensor<4xi64>, window_dilations = dense<1> : tensor<4xi64>}
        : (tensor<64x256x27x27xf32>, tensor<f32>) -> tensor<64x256x13x13xf32>
    return %result : tensor<64x256x13x13xf32>
  }
}

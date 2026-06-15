module {
  func.func @f_20_static_16x1024x13x13_255x1024x1x1_16x255x13x13_1_0_1_correct(%img: tensor<16x1024x13x13xf32>, %ker: tensor<255x1024x1x1xf32>) -> tensor<16x255x13x13xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<16x255x13x13xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<16x255x13x13xf32>) -> tensor<16x255x13x13xf32>
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%img, %ker : tensor<16x1024x13x13xf32>, tensor<255x1024x1x1xf32>)
         outs(%fill : tensor<16x255x13x13xf32>) -> tensor<16x255x13x13xf32>
    return %r : tensor<16x255x13x13xf32>
  }
}

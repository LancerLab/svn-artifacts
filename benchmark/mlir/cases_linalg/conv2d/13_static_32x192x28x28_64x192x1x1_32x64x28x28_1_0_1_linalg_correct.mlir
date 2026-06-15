module {
  func.func @f_13_static_32x192x28x28_64x192x1x1_32x64x28x28_1_0_1_correct(%img: tensor<32x192x28x28xf32>, %ker: tensor<64x192x1x1xf32>) -> tensor<32x64x28x28xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<32x64x28x28xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<32x64x28x28xf32>) -> tensor<32x64x28x28xf32>
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%img, %ker : tensor<32x192x28x28xf32>, tensor<64x192x1x1xf32>)
         outs(%fill : tensor<32x64x28x28xf32>) -> tensor<32x64x28x28xf32>
    return %r : tensor<32x64x28x28xf32>
  }
}

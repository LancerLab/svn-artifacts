module {
  func.func @f_12_efficientnet_64x40x56x56_240x40x1x1_64x240x56x56_1_0_1_correct(%img: tensor<64x40x56x56xf32>, %ker: tensor<240x40x1x1xf32>) -> tensor<64x240x56x56xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<64x240x56x56xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<64x240x56x56xf32>) -> tensor<64x240x56x56xf32>
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%img, %ker : tensor<64x40x56x56xf32>, tensor<240x40x1x1xf32>)
         outs(%fill : tensor<64x240x56x56xf32>) -> tensor<64x240x56x56xf32>
    return %r : tensor<64x240x56x56xf32>
  }
}

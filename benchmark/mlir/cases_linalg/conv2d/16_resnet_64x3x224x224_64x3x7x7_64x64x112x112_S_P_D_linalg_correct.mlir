module {
  func.func @f_16_resnet_64x3x224x224_64x3x7x7_64x64x112x112_S_P_D_correct(%img: tensor<64x3x224x224xf32>, %ker: tensor<64x3x7x7xf32>) -> tensor<64x64x112x112xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<64x64x112x112xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<64x64x112x112xf32>) -> tensor<64x64x112x112xf32>
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%img, %ker : tensor<64x3x224x224xf32>, tensor<64x3x7x7xf32>)
         outs(%fill : tensor<64x64x112x112xf32>) -> tensor<64x64x112x112xf32>
    return %r : tensor<64x64x112x112xf32>
  }
}

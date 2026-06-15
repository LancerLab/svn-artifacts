module {
  func.func @f_9_dynamic_16x64x224x224_128x64x7x7_16x128x112x112_S_P_D_correct(%img: tensor<16x64x224x224xf32>, %ker: tensor<128x64x7x7xf32>) -> tensor<16x128x112x112xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<16x128x112x112xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<16x128x112x112xf32>) -> tensor<16x128x112x112xf32>
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%img, %ker : tensor<16x64x224x224xf32>, tensor<128x64x7x7xf32>)
         outs(%fill : tensor<16x128x112x112xf32>) -> tensor<16x128x112x112xf32>
    return %r : tensor<16x128x112x112xf32>
  }
}

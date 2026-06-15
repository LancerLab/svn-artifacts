module {
  func.func @f_5_dynamic_128xCx112x112_64xCx1x1_128x64x112x112_1_0_1_mismatch(%img: tensor<128x?x112x112xf32>, %ker: tensor<64x?x1x1xf32>) -> tensor<128x64x112x112xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<128x64x112x112xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x64x112x112xf32>) -> tensor<128x64x112x112xf32>
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%img, %ker : tensor<128x?x112x112xf32>, tensor<64x?x1x1xf32>)
         outs(%fill : tensor<128x64x112x112xf32>) -> tensor<128x64x112x112xf32>
    return %r : tensor<128x64x112x112xf32>
  }
}

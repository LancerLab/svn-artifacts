module {
  func.func @f_10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D_mismatch(%img: tensor<32x128x112x112xf32>, %ker: tensor<256x127x3x3xf32>) -> tensor<32x256x56x56xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<32x256x56x56xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<32x256x56x56xf32>) -> tensor<32x256x56x56xf32>
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%img, %ker : tensor<32x128x112x112xf32>, tensor<256x127x3x3xf32>)
         outs(%fill : tensor<32x256x56x56xf32>) -> tensor<32x256x56x56xf32>
    return %r : tensor<32x256x56x56xf32>
  }
}

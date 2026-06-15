module {
  func.func @f_19_vit_32x3x224x224_768x3x16x16_32x768x14x14_16_0_1_mismatch(%img: tensor<32x3x224x224xf32>, %ker: tensor<768x2x16x16xf32>) -> tensor<32x768x14x14xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<32x768x14x14xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<32x768x14x14xf32>) -> tensor<32x768x14x14xf32>
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%img, %ker : tensor<32x3x224x224xf32>, tensor<768x2x16x16xf32>)
         outs(%fill : tensor<32x768x14x14xf32>) -> tensor<32x768x14x14xf32>
    return %r : tensor<32x768x14x14xf32>
  }
}

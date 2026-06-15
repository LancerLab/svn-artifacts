module {
  func.func @f_21_vit_32x768x14x14_1000x768x7x7_32x1000x7x7_S_P_D_correct(%img: tensor<32x768x14x14xf32>, %ker: tensor<1000x768x7x7xf32>) -> tensor<32x1000x7x7xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<32x1000x7x7xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<32x1000x7x7xf32>) -> tensor<32x1000x7x7xf32>
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%img, %ker : tensor<32x768x14x14xf32>, tensor<1000x768x7x7xf32>)
         outs(%fill : tensor<32x1000x7x7xf32>) -> tensor<32x1000x7x7xf32>
    return %r : tensor<32x1000x7x7xf32>
  }
}

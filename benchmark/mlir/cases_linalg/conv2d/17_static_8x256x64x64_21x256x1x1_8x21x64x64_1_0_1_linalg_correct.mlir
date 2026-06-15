module {
  func.func @f_17_static_8x256x64x64_21x256x1x1_8x21x64x64_1_0_1_correct(%img: tensor<8x256x64x64xf32>, %ker: tensor<21x256x1x1xf32>) -> tensor<8x21x64x64xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<8x21x64x64xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<8x21x64x64xf32>) -> tensor<8x21x64x64xf32>
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%img, %ker : tensor<8x256x64x64xf32>, tensor<21x256x1x1xf32>)
         outs(%fill : tensor<8x21x64x64xf32>) -> tensor<8x21x64x64xf32>
    return %r : tensor<8x21x64x64xf32>
  }
}

module {
  func.func @f_7_dynamic_8x256xHxW_512x256x1x1_8x512xHxW_1_0_1_mismatch(%img: tensor<8x256x?x?xf32>, %ker: tensor<512x255x1x1xf32>) -> tensor<8x512x?x?xf32> {
    %cst = arith.constant 0.0 : f32
    %c2 = arith.constant 2 : index
    %h2 = tensor.dim %img, %c2 : tensor<8x256x?x?xf32>
    %c3 = arith.constant 3 : index
    %h3 = tensor.dim %img, %c3 : tensor<8x256x?x?xf32>
    %init = tensor.empty(%h2, %h3) : tensor<8x512x?x?xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<8x512x?x?xf32>) -> tensor<8x512x?x?xf32>
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%img, %ker : tensor<8x256x?x?xf32>, tensor<512x255x1x1xf32>)
         outs(%fill : tensor<8x512x?x?xf32>) -> tensor<8x512x?x?xf32>
    return %r : tensor<8x512x?x?xf32>
  }
}

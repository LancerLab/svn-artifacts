module {
  func.func @f_4_dynamic_32x128xHxW_256x128x3x3_32x256xHxW_S_P_D_mismatch(%img: tensor<32x128x?x?xf32>, %ker: tensor<256x127x3x3xf32>) -> tensor<32x256x?x?xf32> {
    %cst = arith.constant 0.0 : f32
    %c2 = arith.constant 2 : index
    %h2 = tensor.dim %img, %c2 : tensor<32x128x?x?xf32>
    %ks2 = arith.constant 2 : index
    %od2 = arith.subi %h2, %ks2 : index
    %c3 = arith.constant 3 : index
    %h3 = tensor.dim %img, %c3 : tensor<32x128x?x?xf32>
    %ks3 = arith.constant 2 : index
    %od3 = arith.subi %h3, %ks3 : index
    %init = tensor.empty(%od2, %od3) : tensor<32x256x?x?xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<32x256x?x?xf32>) -> tensor<32x256x?x?xf32>
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%img, %ker : tensor<32x128x?x?xf32>, tensor<256x127x3x3xf32>)
         outs(%fill : tensor<32x256x?x?xf32>) -> tensor<32x256x?x?xf32>
    return %r : tensor<32x256x?x?xf32>
  }
}

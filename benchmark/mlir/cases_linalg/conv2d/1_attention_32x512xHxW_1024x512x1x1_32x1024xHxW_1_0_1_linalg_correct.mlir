module {
  func.func @f_1_attention_32x512xHxW_1024x512x1x1_32x1024xHxW_1_0_1_correct(%img: tensor<32x512x?x?xf32>, %ker: tensor<1024x512x1x1xf32>) -> tensor<32x1024x?x?xf32> {
    %cst = arith.constant 0.0 : f32
    %c2 = arith.constant 2 : index
    %h2 = tensor.dim %img, %c2 : tensor<32x512x?x?xf32>
    %c3 = arith.constant 3 : index
    %h3 = tensor.dim %img, %c3 : tensor<32x512x?x?xf32>
    %init = tensor.empty(%h2, %h3) : tensor<32x1024x?x?xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<32x1024x?x?xf32>) -> tensor<32x1024x?x?xf32>
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%img, %ker : tensor<32x512x?x?xf32>, tensor<1024x512x1x1xf32>)
         outs(%fill : tensor<32x1024x?x?xf32>) -> tensor<32x1024x?x?xf32>
    return %r : tensor<32x1024x?x?xf32>
  }
}

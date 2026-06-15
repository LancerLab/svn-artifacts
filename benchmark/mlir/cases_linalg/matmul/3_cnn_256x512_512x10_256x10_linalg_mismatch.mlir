module {
  func.func @f_3_cnn_256x512_512x10_256x10_mismatch(%a: tensor<256x512xf32>, %b: tensor<511x10xf32>) -> tensor<256x10xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<256x10xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<256x10xf32>) -> tensor<256x10xf32>
    %r = linalg.matmul ins(%a, %b : tensor<256x512xf32>, tensor<511x10xf32>)
                        outs(%fill : tensor<256x10xf32>) -> tensor<256x10xf32>
    return %r : tensor<256x10xf32>
  }
}

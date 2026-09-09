module {
  func.func @f_3_cnn_256x512_512x10_256x10(%lhs: tensor<256x512xf32>, %rhs: tensor<512x10xf32>) -> tensor<256x10xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lhs_d0 = tensor.dim %lhs, %c0 : tensor<256x512xf32>
    %lhs_d1 = tensor.dim %lhs, %c1 : tensor<256x512xf32>
    %rhs_d0 = tensor.dim %rhs, %c0 : tensor<512x10xf32>
    %rhs_d1 = tensor.dim %rhs, %c1 : tensor<512x10xf32>
    %zero = arith.constant 0.0 : f32
    %out = tensor.empty() : tensor<256x10xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%out : tensor<256x10xf32>) -> tensor<256x10xf32>
    %result = linalg.matmul ins(%lhs, %rhs : tensor<256x512xf32>, tensor<512x10xf32>) outs(%filled : tensor<256x10xf32>) -> tensor<256x10xf32>
    return %result : tensor<256x10xf32>
  }
}

module {
  func.func @f_10_dynamic_128x1280_1280xN_128xN(%lhs: tensor<128x1280xf32>, %rhs: tensor<1280x?xf32>) -> tensor<128x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lhs_d0 = tensor.dim %lhs, %c0 : tensor<128x1280xf32>
    %lhs_d1 = tensor.dim %lhs, %c1 : tensor<128x1280xf32>
    %rhs_d0 = tensor.dim %rhs, %c0 : tensor<1280x?xf32>
    %rhs_d1 = tensor.dim %rhs, %c1 : tensor<1280x?xf32>
    %zero = arith.constant 0.0 : f32
    %out = tensor.empty(%rhs_d1) : tensor<128x?xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%out : tensor<128x?xf32>) -> tensor<128x?xf32>
    %result = linalg.matmul ins(%lhs, %rhs : tensor<128x1280xf32>, tensor<1280x?xf32>) outs(%filled : tensor<128x?xf32>) -> tensor<128x?xf32>
    return %result : tensor<128x?xf32>
  }
}

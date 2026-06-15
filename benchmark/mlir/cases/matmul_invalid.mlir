module {
  func.func @matmul_invalid(%lhs: tensor<4x8xf32>, %rhs: tensor<7x6xf32>) -> tensor<4x6xf32> {
    %init = tensor.empty() : tensor<4x6xf32>
    %result = linalg.matmul ins(%lhs, %rhs : tensor<4x8xf32>, tensor<7x6xf32>)
                            outs(%init : tensor<4x6xf32>) -> tensor<4x6xf32>
    return %result : tensor<4x6xf32>
  }
}

module {
  func.func @f_5_dynamic_Bx2048_2048x1000_Bx1000(%lhs: tensor<?x2048xf32>, %rhs: tensor<2048x1000xf32>) -> tensor<?x1000xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lhs_d0 = tensor.dim %lhs, %c0 : tensor<?x2048xf32>
    %lhs_d1 = tensor.dim %lhs, %c1 : tensor<?x2048xf32>
    %rhs_d0 = tensor.dim %rhs, %c0 : tensor<2048x1000xf32>
    %rhs_d1 = tensor.dim %rhs, %c1 : tensor<2048x1000xf32>
    %zero = arith.constant 0.0 : f32
    %out = tensor.empty(%lhs_d0) : tensor<?x1000xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%out : tensor<?x1000xf32>) -> tensor<?x1000xf32>
    %result = linalg.matmul ins(%lhs, %rhs : tensor<?x2048xf32>, tensor<2048x1000xf32>) outs(%filled : tensor<?x1000xf32>) -> tensor<?x1000xf32>
    return %result : tensor<?x1000xf32>
  }
}

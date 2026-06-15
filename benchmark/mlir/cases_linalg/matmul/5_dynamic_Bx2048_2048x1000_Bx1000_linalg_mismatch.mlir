module {
  func.func @f_5_dynamic_Bx2048_2048x1000_Bx1000_mismatch(%a: tensor<?x2048xf32>, %b: tensor<2047x1000xf32>) -> tensor<?x1000xf32> {
    %cst = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %m = tensor.dim %a, %c0 : tensor<?x2048xf32>
    %init = tensor.empty(%m) : tensor<?x1000xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x1000xf32>) -> tensor<?x1000xf32>
    %r = linalg.matmul ins(%a, %b : tensor<?x2048xf32>, tensor<2047x1000xf32>)
                        outs(%fill : tensor<?x1000xf32>) -> tensor<?x1000xf32>
    return %r : tensor<?x1000xf32>
  }
}

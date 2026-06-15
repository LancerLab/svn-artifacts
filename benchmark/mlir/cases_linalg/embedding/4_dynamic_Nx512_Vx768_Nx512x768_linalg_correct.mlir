module {
  func.func @f_4_dynamic_Nx512_Vx768_Nx512x768_correct(%indices: tensor<?x512xi64>, %weights: tensor<?x768xf32>) -> tensor<?x512x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ci0 = arith.constant 0 : index
    %od0 = tensor.dim %indices, %ci0 : tensor<?x512xi64>
    %init = tensor.empty(%od0) : tensor<?x512x768xf32>
    %cst = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x512x768xf32>) -> tensor<?x512x768xf32>
    return %fill : tensor<?x512x768xf32>
  }
}

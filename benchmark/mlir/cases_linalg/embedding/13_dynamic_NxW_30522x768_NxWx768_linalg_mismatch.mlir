module {
  func.func @f_13_dynamic_NxW_30522x768_NxWx768_mismatch(%indices: tensor<?x?xi64>, %weights: tensor<30522x767xf32>) -> tensor<?x?x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ci0 = arith.constant 0 : index
    %od0 = tensor.dim %indices, %ci0 : tensor<?x?xi64>
    %ci1 = arith.constant 1 : index
    %od1 = tensor.dim %indices, %ci1 : tensor<?x?xi64>
    %init = tensor.empty(%od0, %od1) : tensor<?x?x768xf32>
    %cst = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?x768xf32>) -> tensor<?x?x768xf32>
    return %fill : tensor<?x?x768xf32>
  }
}

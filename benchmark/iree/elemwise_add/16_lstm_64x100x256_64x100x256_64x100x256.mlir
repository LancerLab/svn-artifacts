module {
  func.func @f_16_lstm_64x100x256_64x100x256_64x100x256(%lhs: tensor<64x100x256xf32>, %rhs: tensor<64x100x256xf32>) -> tensor<64x100x256xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %lhs_d0 = tensor.dim %lhs, %c0 : tensor<64x100x256xf32>
    %lhs_d1 = tensor.dim %lhs, %c1 : tensor<64x100x256xf32>
    %lhs_d2 = tensor.dim %lhs, %c2 : tensor<64x100x256xf32>
    %rhs_d0 = tensor.dim %rhs, %c0 : tensor<64x100x256xf32>
    %rhs_d1 = tensor.dim %rhs, %c1 : tensor<64x100x256xf32>
    %rhs_d2 = tensor.dim %rhs, %c2 : tensor<64x100x256xf32>
    %out = tensor.empty() : tensor<64x100x256xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%lhs, %rhs : tensor<64x100x256xf32>, tensor<64x100x256xf32>) outs(%out : tensor<64x100x256xf32>) {
    ^bb0(%a: f32, %b: f32, %init: f32):
      %res = arith.addf %a, %b : f32
      linalg.yield %res : f32
    } -> tensor<64x100x256xf32>
    return %result : tensor<64x100x256xf32>
  }
}

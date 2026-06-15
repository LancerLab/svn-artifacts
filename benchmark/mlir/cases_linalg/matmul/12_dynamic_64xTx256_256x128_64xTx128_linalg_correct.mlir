#map_lhs = affine_map<(b, m, n, k) -> (b, m, k)>
#map_rhs = affine_map<(b, m, n, k) -> (k, n)>
#map_out = affine_map<(b, m, n, k) -> (b, m, n)>

module {
  func.func @f_12_dynamic_64xTx256_256x128_64xTx128_correct(%a: tensor<64x?x256xf32>, %b: tensor<256x128xf32>) -> tensor<64x?x128xf32> {
    %cst = arith.constant 0.0 : f32
    %c_i0 = arith.constant 1 : index
    %d1 = tensor.dim %a, %c_i0 : tensor<64x?x256xf32>
    %init = tensor.empty(%d1) : tensor<64x?x128xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<64x?x128xf32>) -> tensor<64x?x128xf32>
    %r = linalg.generic {
      indexing_maps = [#map_lhs, #map_rhs, #map_out],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<64x?x256xf32>, tensor<256x128xf32>)
      outs(%fill : tensor<64x?x128xf32>) {
    ^bb0(%in_a: f32, %in_b: f32, %acc: f32):
      %mul = arith.mulf %in_a, %in_b : f32
      %add = arith.addf %acc, %mul : f32
      linalg.yield %add : f32
    } -> tensor<64x?x128xf32>
    return %r : tensor<64x?x128xf32>
  }
}

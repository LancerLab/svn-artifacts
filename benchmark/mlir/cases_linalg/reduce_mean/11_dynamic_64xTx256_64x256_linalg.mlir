module {
  func.func @f_11_dynamic_64xTx256_64x256_linalg(%input: tensor<64x?x256xf32>) -> tensor<64x256xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<64x256xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<64x256xf32>) -> tensor<64x256xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%input : tensor<64x?x256xf32>) outs(%fill : tensor<64x256xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %sum = arith.addf %acc, %in : f32
      linalg.yield %sum : f32
    } -> tensor<64x256xf32>
    return %r : tensor<64x256xf32>
  }
}

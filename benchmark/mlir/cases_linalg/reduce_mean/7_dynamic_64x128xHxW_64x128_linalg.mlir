module {
  func.func @f_7_dynamic_64x128xHxW_64x128_linalg(%input: tensor<64x128x?x?xf32>) -> tensor<64x128xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<64x128xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<64x128xf32>) -> tensor<64x128xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction", "parallel"]
    } ins(%input : tensor<64x128x?x?xf32>) outs(%fill : tensor<64x128xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %sum = arith.addf %acc, %in : f32
      linalg.yield %sum : f32
    } -> tensor<64x128xf32>
    return %r : tensor<64x128xf32>
  }
}

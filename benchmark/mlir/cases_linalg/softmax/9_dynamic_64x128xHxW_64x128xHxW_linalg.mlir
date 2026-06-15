module {
  func.func @f_9_dynamic_64x128xHxW_64x128xHxW_linalg(%input: tensor<64x128x?x?xf32>) -> tensor<64x128x?x?xf32> {
    %c2 = arith.constant 2 : index
    %d2 = tensor.dim %input, %c2 : tensor<64x128x?x?xf32>
    %c3 = arith.constant 3 : index
    %d3 = tensor.dim %input, %c3 : tensor<64x128x?x?xf32>
    %init = tensor.empty(%d2, %d3) : tensor<64x128x?x?xf32>
    %cst = arith.constant 0.0 : f32
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input : tensor<64x128x?x?xf32>) outs(%init : tensor<64x128x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %abs = math.absf %in : f32
      linalg.yield %abs : f32
    } -> tensor<64x128x?x?xf32>
    return %r : tensor<64x128x?x?xf32>
  }
}

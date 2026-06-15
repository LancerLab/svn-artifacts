module {
  func.func @f_7_dynamic_32x197xE_32x197xE_linalg(%input: tensor<32x197x?xf32>) -> tensor<32x197x?xf32> {
    %c2 = arith.constant 2 : index
    %d2 = tensor.dim %input, %c2 : tensor<32x197x?xf32>
    %init = tensor.empty(%d2) : tensor<32x197x?xf32>
    %cst = arith.constant 0.0 : f32
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input : tensor<32x197x?xf32>) outs(%init : tensor<32x197x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %abs = math.absf %in : f32
      linalg.yield %abs : f32
    } -> tensor<32x197x?xf32>
    return %r : tensor<32x197x?xf32>
  }
}

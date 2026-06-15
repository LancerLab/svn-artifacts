module {
  func.func @f_1_bert_32x512x768_32x512x768_linalg(%input: tensor<32x512x768xf32>) -> tensor<32x512x768xf32> {
    %init = tensor.empty() : tensor<32x512x768xf32>
    %cst = arith.constant 0.0 : f32
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input : tensor<32x512x768xf32>) outs(%init : tensor<32x512x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %abs = math.absf %in : f32
      linalg.yield %abs : f32
    } -> tensor<32x512x768xf32>
    return %r : tensor<32x512x768xf32>
  }
}

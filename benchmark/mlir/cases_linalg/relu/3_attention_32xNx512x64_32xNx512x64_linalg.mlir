module {
  func.func @f_3_attention_32xNx512x64_32xNx512x64_linalg(%input: tensor<32x?x512x64xf32>) -> tensor<32x?x512x64xf32> {
    %c1 = arith.constant 1 : index
    %d1 = tensor.dim %input, %c1 : tensor<32x?x512x64xf32>
    %init = tensor.empty(%d1) : tensor<32x?x512x64xf32>
    %cst = arith.constant 0.0 : f32
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input : tensor<32x?x512x64xf32>) outs(%init : tensor<32x?x512x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %abs = math.absf %in : f32
      linalg.yield %abs : f32
    } -> tensor<32x?x512x64xf32>
    return %r : tensor<32x?x512x64xf32>
  }
}

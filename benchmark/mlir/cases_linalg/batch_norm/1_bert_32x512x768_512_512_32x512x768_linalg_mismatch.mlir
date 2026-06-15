module {
  func.func @f_1_bert_32x512x768_512_512_32x512x768_mismatch(%input: tensor<32x512x768xf32>, %gamma: tensor<511xf32>, %beta: tensor<512xf32>) -> tensor<32x512x768xf32> {
    %init = tensor.empty() : tensor<32x512x768xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1)>, affine_map<(d0, d1, d2) -> (d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input, %gamma, %beta : tensor<32x512x768xf32>, tensor<511xf32>, tensor<512xf32>) outs(%init : tensor<32x512x768xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %out: f32):
      %scaled = arith.mulf %x, %g : f32
      %res = arith.addf %scaled, %b : f32
      linalg.yield %res : f32
    } -> tensor<32x512x768xf32>
    return %r : tensor<32x512x768xf32>
  }
}

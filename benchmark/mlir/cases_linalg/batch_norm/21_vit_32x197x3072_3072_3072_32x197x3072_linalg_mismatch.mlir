module {
  func.func @f_21_vit_32x197x3072_3072_3072_32x197x3072_mismatch(%input: tensor<32x197x3072xf32>, %gamma: tensor<3071xf32>, %beta: tensor<3072xf32>) -> tensor<32x197x3072xf32> {
    %init = tensor.empty() : tensor<32x197x3072xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input, %gamma, %beta : tensor<32x197x3072xf32>, tensor<3071xf32>, tensor<3072xf32>) outs(%init : tensor<32x197x3072xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %out: f32):
      %scaled = arith.mulf %x, %g : f32
      %res = arith.addf %scaled, %b : f32
      linalg.yield %res : f32
    } -> tensor<32x197x3072xf32>
    return %r : tensor<32x197x3072xf32>
  }
}

module {
  func.func @f_19_transformer_32x512x2048_2048_2048_32x512x2048_correct(%input: tensor<32x512x2048xf32>, %gamma: tensor<2048xf32>, %beta: tensor<2048xf32>) -> tensor<32x512x2048xf32> {
    %init = tensor.empty() : tensor<32x512x2048xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input, %gamma, %beta : tensor<32x512x2048xf32>, tensor<2048xf32>, tensor<2048xf32>) outs(%init : tensor<32x512x2048xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %out: f32):
      %scaled = arith.mulf %x, %g : f32
      %res = arith.addf %scaled, %b : f32
      linalg.yield %res : f32
    } -> tensor<32x512x2048xf32>
    return %r : tensor<32x512x2048xf32>
  }
}

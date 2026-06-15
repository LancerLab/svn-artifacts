module {
  func.func @f_15_gpt_16x1024x4096_4096_4096_16x1024x4096_correct(%input: tensor<16x1024x4096xf32>, %gamma: tensor<4096xf32>, %beta: tensor<4096xf32>) -> tensor<16x1024x4096xf32> {
    %init = tensor.empty() : tensor<16x1024x4096xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input, %gamma, %beta : tensor<16x1024x4096xf32>, tensor<4096xf32>, tensor<4096xf32>) outs(%init : tensor<16x1024x4096xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %out: f32):
      %scaled = arith.mulf %x, %g : f32
      %res = arith.addf %scaled, %b : f32
      linalg.yield %res : f32
    } -> tensor<16x1024x4096xf32>
    return %r : tensor<16x1024x4096xf32>
  }
}

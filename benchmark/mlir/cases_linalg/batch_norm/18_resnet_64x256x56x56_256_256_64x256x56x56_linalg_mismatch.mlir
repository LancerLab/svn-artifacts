module {
  func.func @f_18_resnet_64x256x56x56_256_256_64x256x56x56_mismatch(%input: tensor<64x256x56x56xf32>, %gamma: tensor<255xf32>, %beta: tensor<256xf32>) -> tensor<64x256x56x56xf32> {
    %init = tensor.empty() : tensor<64x256x56x56xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input, %gamma, %beta : tensor<64x256x56x56xf32>, tensor<255xf32>, tensor<256xf32>) outs(%init : tensor<64x256x56x56xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %out: f32):
      %scaled = arith.mulf %x, %g : f32
      %res = arith.addf %scaled, %b : f32
      linalg.yield %res : f32
    } -> tensor<64x256x56x56xf32>
    return %r : tensor<64x256x56x56xf32>
  }
}

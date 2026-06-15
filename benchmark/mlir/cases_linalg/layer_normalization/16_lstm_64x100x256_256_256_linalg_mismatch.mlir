module {
  func.func @f_16_lstm_64x100x256_256_256_mismatch(%input: tensor<64x100x256xf32>, %gamma: tensor<255xf32>, %beta: tensor<255xf32>) -> tensor<64x100x256xf32> {
    %init = tensor.empty() : tensor<64x100x256xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input, %gamma, %beta : tensor<64x100x256xf32>, tensor<255xf32>, tensor<255xf32>) outs(%init : tensor<64x100x256xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %out: f32):
      %scaled = arith.mulf %x, %g : f32
      %res = arith.addf %scaled, %b : f32
      linalg.yield %res : f32
    } -> tensor<64x100x256xf32>
    return %r : tensor<64x100x256xf32>
  }
}

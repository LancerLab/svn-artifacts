module {
  func.func @f_16_lstm_64x100x256_linalg(%input: tensor<64x100x256xf32>) -> tensor<64x100x256xf32> {
    %init = tensor.empty() : tensor<64x100x256xf32>
    %cst = arith.constant 0.0 : f32
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input : tensor<64x100x256xf32>) outs(%init : tensor<64x100x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %abs = math.absf %in : f32
      linalg.yield %abs : f32
    } -> tensor<64x100x256xf32>
    return %r : tensor<64x100x256xf32>
  }
}

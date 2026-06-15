module {
  func.func @f_6_dynamic_128xCx112x112_112x112_112x112_correct(%input: tensor<128x?x112x112xf32>, %gamma: tensor<112x112xf32>, %beta: tensor<112x112xf32>) -> tensor<128x?x112x112xf32> {
    %c1 = arith.constant 1 : index
    %d1 = tensor.dim %input, %c1 : tensor<128x?x112x112xf32>
    %init = tensor.empty(%d1) : tensor<128x?x112x112xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input, %gamma, %beta : tensor<128x?x112x112xf32>, tensor<112x112xf32>, tensor<112x112xf32>) outs(%init : tensor<128x?x112x112xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %out: f32):
      %scaled = arith.mulf %x, %g : f32
      %res = arith.addf %scaled, %b : f32
      linalg.yield %res : f32
    } -> tensor<128x?x112x112xf32>
    return %r : tensor<128x?x112x112xf32>
  }
}

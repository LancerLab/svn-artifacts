module {
  func.func @f_9_dynamic_64x128xHxW_128_128_64x128xHxW_correct(%input: tensor<64x128x?x?xf32>, %gamma: tensor<128xf32>, %beta: tensor<128xf32>) -> tensor<64x128x?x?xf32> {
    %c2 = arith.constant 2 : index
    %d2 = tensor.dim %input, %c2 : tensor<64x128x?x?xf32>
    %c3 = arith.constant 3 : index
    %d3 = tensor.dim %input, %c3 : tensor<64x128x?x?xf32>
    %init = tensor.empty(%d2, %d3) : tensor<64x128x?x?xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input, %gamma, %beta : tensor<64x128x?x?xf32>, tensor<128xf32>, tensor<128xf32>) outs(%init : tensor<64x128x?x?xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %out: f32):
      %scaled = arith.mulf %x, %g : f32
      %res = arith.addf %scaled, %b : f32
      linalg.yield %res : f32
    } -> tensor<64x128x?x?xf32>
    return %r : tensor<64x128x?x?xf32>
  }
}

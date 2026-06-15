module {
  func.func @f_10_dynamic_16x512xHxW_HxW_HxW_correct(%input: tensor<16x512x?x?xf32>, %gamma: tensor<?x?xf32>, %beta: tensor<?x?xf32>) -> tensor<16x512x?x?xf32> {
    %c2 = arith.constant 2 : index
    %d2 = tensor.dim %input, %c2 : tensor<16x512x?x?xf32>
    %c3 = arith.constant 3 : index
    %d3 = tensor.dim %input, %c3 : tensor<16x512x?x?xf32>
    %init = tensor.empty(%d2, %d3) : tensor<16x512x?x?xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input, %gamma, %beta : tensor<16x512x?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%init : tensor<16x512x?x?xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %out: f32):
      %scaled = arith.mulf %x, %g : f32
      %res = arith.addf %scaled, %b : f32
      linalg.yield %res : f32
    } -> tensor<16x512x?x?xf32>
    return %r : tensor<16x512x?x?xf32>
  }
}

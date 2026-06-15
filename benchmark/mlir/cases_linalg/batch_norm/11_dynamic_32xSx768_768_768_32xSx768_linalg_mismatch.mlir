module {
  func.func @f_11_dynamic_32xSx768_768_768_32xSx768_mismatch(%input: tensor<32x?x768xf32>, %gamma: tensor<767xf32>, %beta: tensor<768xf32>) -> tensor<32x?x768xf32> {
    %c1 = arith.constant 1 : index
    %d1 = tensor.dim %input, %c1 : tensor<32x?x768xf32>
    %init = tensor.empty(%d1) : tensor<32x?x768xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input, %gamma, %beta : tensor<32x?x768xf32>, tensor<767xf32>, tensor<768xf32>) outs(%init : tensor<32x?x768xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %out: f32):
      %scaled = arith.mulf %x, %g : f32
      %res = arith.addf %scaled, %b : f32
      linalg.yield %res : f32
    } -> tensor<32x?x768xf32>
    return %r : tensor<32x?x768xf32>
  }
}

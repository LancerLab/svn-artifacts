module {
  func.func @f_9_dynamic_32xPx768_32x768_linalg(%input: tensor<32x?x768xf32>) -> tensor<32x768xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<32x768xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<32x768xf32>) -> tensor<32x768xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%input : tensor<32x?x768xf32>) outs(%fill : tensor<32x768xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %sum = arith.addf %acc, %in : f32
      linalg.yield %sum : f32
    } -> tensor<32x768xf32>
    return %r : tensor<32x768xf32>
  }
}

module {
  func.func @f_5_dynamic_32xCx112x112_32x112x112_linalg(%input: tensor<32x?x112x112xf32>) -> tensor<32x112x112xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<32x112x112xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<32x112x112xf32>) -> tensor<32x112x112xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>],
      iterator_types = ["parallel", "reduction", "parallel", "parallel"]
    } ins(%input : tensor<32x?x112x112xf32>) outs(%fill : tensor<32x112x112xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %sum = arith.addf %acc, %in : f32
      linalg.yield %sum : f32
    } -> tensor<32x112x112xf32>
    return %r : tensor<32x112x112xf32>
  }
}

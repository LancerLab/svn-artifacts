module {
  func.func @f_14_gpt_16x1024x1024_16x1024_linalg(%input: tensor<16x1024x1024xf32>) -> tensor<16x1024xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<16x1024xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<16x1024xf32>) -> tensor<16x1024xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%input : tensor<16x1024x1024xf32>) outs(%fill : tensor<16x1024xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %sum = arith.addf %acc, %in : f32
      linalg.yield %sum : f32
    } -> tensor<16x1024xf32>
    return %r : tensor<16x1024xf32>
  }
}

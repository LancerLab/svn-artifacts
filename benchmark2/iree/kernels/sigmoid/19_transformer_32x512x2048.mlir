module {
  func.func @f_19_transformer_32x512x2048(%input: tensor<32x512x2048xf32>) -> tensor<32x512x2048xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<32x512x2048xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<32x512x2048xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<32x512x2048xf32>
    %out = tensor.empty() : tensor<32x512x2048xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input : tensor<32x512x2048xf32>) outs(%out : tensor<32x512x2048xf32>) {
    ^bb0(%in: f32, %init: f32):
      %neg = arith.negf %in : f32
      %expv = math.exp %neg : f32
      %one = arith.constant 1.0 : f32
      %den = arith.addf %one, %expv : f32
      %res = arith.divf %one, %den : f32
      linalg.yield %res : f32
    } -> tensor<32x512x2048xf32>
    return %result : tensor<32x512x2048xf32>
  }
}

module {
  func.func @f_8_dynamic_16x1024xD(%input: tensor<16x1024x?xf32>) -> tensor<16x1024x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<16x1024x?xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<16x1024x?xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<16x1024x?xf32>
    %out = tensor.empty(%input_d2) : tensor<16x1024x?xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input : tensor<16x1024x?xf32>) outs(%out : tensor<16x1024x?xf32>) {
    ^bb0(%in: f32, %init: f32):
      %neg = arith.negf %in : f32
      %expv = math.exp %neg : f32
      %one = arith.constant 1.0 : f32
      %den = arith.addf %one, %expv : f32
      %res = arith.divf %one, %den : f32
      linalg.yield %res : f32
    } -> tensor<16x1024x?xf32>
    return %result : tensor<16x1024x?xf32>
  }
}

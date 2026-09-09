module {
  func.func @f_15_gpt_16x1024x4096(%input: tensor<16x1024x4096xf32>) -> tensor<16x1024x4096xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<16x1024x4096xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<16x1024x4096xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<16x1024x4096xf32>
    %out = tensor.empty() : tensor<16x1024x4096xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input : tensor<16x1024x4096xf32>) outs(%out : tensor<16x1024x4096xf32>) {
    ^bb0(%in: f32, %init: f32):
      %half   = arith.constant 0.5 : f32
      %isqrt2 = arith.constant 0.7071067811865476 : f32
      %sc     = arith.mulf %in, %isqrt2 : f32
      %efv    = math.erf %sc : f32
      %one    = arith.constant 1.0 : f32
      %ep1    = arith.addf %efv, %one : f32
      %hx     = arith.mulf %in, %half : f32
      %res    = arith.mulf %hx, %ep1 : f32
      linalg.yield %res : f32
    } -> tensor<16x1024x4096xf32>
    return %result : tensor<16x1024x4096xf32>
  }
}

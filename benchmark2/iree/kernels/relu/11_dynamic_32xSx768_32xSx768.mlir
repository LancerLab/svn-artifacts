module {
  func.func @f_11_dynamic_32xSx768_32xSx768(%input: tensor<32x?x768xf32>) -> tensor<32x?x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<32x?x768xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<32x?x768xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<32x?x768xf32>
    %out = tensor.empty(%input_d1) : tensor<32x?x768xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input : tensor<32x?x768xf32>) outs(%out : tensor<32x?x768xf32>) {
    ^bb0(%in: f32, %init: f32):
      %zero = arith.constant 0.0 : f32
      %res = arith.maximumf %in, %zero : f32
      linalg.yield %res : f32
    } -> tensor<32x?x768xf32>
    return %result : tensor<32x?x768xf32>
  }
}

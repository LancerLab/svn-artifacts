module {
  func.func @bench_relu(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<?x?xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<?x?xf32>
    %out = tensor.empty(%input_d0, %input_d1) : tensor<?x?xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : tensor<?x?xf32>) outs(%out : tensor<?x?xf32>) {
    ^bb0(%in: f32, %init: f32):
      %zero = arith.constant 0.0 : f32
      %res = arith.maximumf %in, %zero : f32
      linalg.yield %res : f32
    } -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }
}

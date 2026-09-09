module {
  func.func @f_4_dynamic_Nx256x56x56_256_256_Nx256x56x56(%input: tensor<?x256x56x56xf32>, %gamma: tensor<256xf32>, %beta: tensor<256xf32>) -> tensor<?x256x56x56xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<?x256x56x56xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<?x256x56x56xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<?x256x56x56xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<?x256x56x56xf32>
    %out = tensor.empty(%input_d0) : tensor<?x256x56x56xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input, %gamma, %beta : tensor<?x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>) outs(%out : tensor<?x256x56x56xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %init: f32):
      %sc  = arith.mulf %x, %g : f32
      %res = arith.addf %sc, %b : f32
      linalg.yield %res : f32
    } -> tensor<?x256x56x56xf32>
    return %result : tensor<?x256x56x56xf32>
  }
}

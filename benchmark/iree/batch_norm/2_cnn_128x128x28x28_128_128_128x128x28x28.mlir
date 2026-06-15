module {
  func.func @f_2_cnn_128x128x28x28_128_128_128x128x28x28(%input: tensor<128x128x28x28xf32>, %gamma: tensor<128xf32>, %beta: tensor<128xf32>) -> tensor<128x128x28x28xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<128x128x28x28xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<128x128x28x28xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<128x128x28x28xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<128x128x28x28xf32>
    %out = tensor.empty() : tensor<128x128x28x28xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input, %gamma, %beta : tensor<128x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) outs(%out : tensor<128x128x28x28xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %init: f32):
      %sc  = arith.mulf %x, %g : f32
      %res = arith.addf %sc, %b : f32
      linalg.yield %res : f32
    } -> tensor<128x128x28x28xf32>
    return %result : tensor<128x128x28x28xf32>
  }
}

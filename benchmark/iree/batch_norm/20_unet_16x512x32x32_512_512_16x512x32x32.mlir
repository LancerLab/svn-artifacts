module {
  func.func @f_20_unet_16x512x32x32_512_512_16x512x32x32(%input: tensor<16x512x32x32xf32>, %gamma: tensor<512xf32>, %beta: tensor<512xf32>) -> tensor<16x512x32x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<16x512x32x32xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<16x512x32x32xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<16x512x32x32xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<16x512x32x32xf32>
    %out = tensor.empty() : tensor<16x512x32x32xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input, %gamma, %beta : tensor<16x512x32x32xf32>, tensor<512xf32>, tensor<512xf32>) outs(%out : tensor<16x512x32x32xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %init: f32):
      %sc  = arith.mulf %x, %g : f32
      %res = arith.addf %sc, %b : f32
      linalg.yield %res : f32
    } -> tensor<16x512x32x32xf32>
    return %result : tensor<16x512x32x32xf32>
  }
}

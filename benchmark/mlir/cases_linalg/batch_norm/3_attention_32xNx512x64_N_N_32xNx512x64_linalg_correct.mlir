module {
  func.func @f_3_attention_32xNx512x64_N_N_32xNx512x64_correct(%input: tensor<32x?x512x64xf32>, %gamma: tensor<?xf32>, %beta: tensor<?xf32>) -> tensor<32x?x512x64xf32> {
    %c1 = arith.constant 1 : index
    %d1 = tensor.dim %input, %c1 : tensor<32x?x512x64xf32>
    %init = tensor.empty(%d1) : tensor<32x?x512x64xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%input, %gamma, %beta : tensor<32x?x512x64xf32>, tensor<?xf32>, tensor<?xf32>) outs(%init : tensor<32x?x512x64xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %out: f32):
      %scaled = arith.mulf %x, %g : f32
      %res = arith.addf %scaled, %b : f32
      linalg.yield %res : f32
    } -> tensor<32x?x512x64xf32>
    return %r : tensor<32x?x512x64xf32>
  }
}

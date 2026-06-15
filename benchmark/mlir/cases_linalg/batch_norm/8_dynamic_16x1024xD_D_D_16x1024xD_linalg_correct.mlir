module {
  func.func @f_8_dynamic_16x1024xD_D_D_16x1024xD_correct(%input: tensor<16x1024x?xf32>, %gamma: tensor<?xf32>, %beta: tensor<?xf32>) -> tensor<16x1024x?xf32> {
    %c2 = arith.constant 2 : index
    %d2 = tensor.dim %input, %c2 : tensor<16x1024x?xf32>
    %init = tensor.empty(%d2) : tensor<16x1024x?xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input, %gamma, %beta : tensor<16x1024x?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%init : tensor<16x1024x?xf32>) {
    ^bb0(%x: f32, %g: f32, %b: f32, %out: f32):
      %scaled = arith.mulf %x, %g : f32
      %res = arith.addf %scaled, %b : f32
      linalg.yield %res : f32
    } -> tensor<16x1024x?xf32>
    return %r : tensor<16x1024x?xf32>
  }
}

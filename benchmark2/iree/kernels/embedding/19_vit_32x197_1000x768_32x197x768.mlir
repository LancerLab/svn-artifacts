module {
  func.func @f_19_vit_32x197_1000x768_32x197x768(%indices: tensor<32x197xi64>, %table: tensor<1000x768xf32>) -> tensor<32x197x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %indices_d0 = tensor.dim %indices, %c0 : tensor<32x197xi64>
    %indices_d1 = tensor.dim %indices, %c1 : tensor<32x197xi64>
    %table_d0 = tensor.dim %table, %c0 : tensor<1000x768xf32>
    %table_d1 = tensor.dim %table, %c1 : tensor<1000x768xf32>
    %out = tensor.empty() : tensor<32x197x768xf32>
    %zero_emb = arith.constant 0.0 : f32
    %out_init = linalg.fill ins(%zero_emb : f32) outs(%out : tensor<32x197x768xf32>) -> tensor<32x197x768xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%indices : tensor<32x197xi64>) outs(%out_init : tensor<32x197x768xf32>) {
    ^bb0(%raw_idx: i64, %init: f32):
      %ei0 = linalg.index 2 : index
      %row = arith.index_cast %raw_idx : i64 to index
      %val = tensor.extract %table[%row, %ei0] : tensor<1000x768xf32>
      linalg.yield %val : f32
    } -> tensor<32x197x768xf32>
    return %result : tensor<32x197x768xf32>
  }
}

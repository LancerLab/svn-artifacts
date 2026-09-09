module {
  func.func @f_4_dynamic_Nx512_Vx768_Nx512x768(%indices: tensor<?x512xi64>, %table: tensor<?x768xf32>) -> tensor<?x512x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %indices_d0 = tensor.dim %indices, %c0 : tensor<?x512xi64>
    %indices_d1 = tensor.dim %indices, %c1 : tensor<?x512xi64>
    %table_d0 = tensor.dim %table, %c0 : tensor<?x768xf32>
    %table_d1 = tensor.dim %table, %c1 : tensor<?x768xf32>
    %out = tensor.empty(%indices_d0) : tensor<?x512x768xf32>
    %zero_emb = arith.constant 0.0 : f32
    %out_init = linalg.fill ins(%zero_emb : f32) outs(%out : tensor<?x512x768xf32>) -> tensor<?x512x768xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%indices : tensor<?x512xi64>) outs(%out_init : tensor<?x512x768xf32>) {
    ^bb0(%raw_idx: i64, %init: f32):
      %ei0 = linalg.index 2 : index
      %row = arith.index_cast %raw_idx : i64 to index
      %val = tensor.extract %table[%row, %ei0] : tensor<?x768xf32>
      linalg.yield %val : f32
    } -> tensor<?x512x768xf32>
    return %result : tensor<?x512x768xf32>
  }
}

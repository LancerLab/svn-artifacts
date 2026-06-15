module {
  func.func @f_16_general_64x256_20000x512_64x256x512(%indices: tensor<64x256xi64>, %table: tensor<20000x512xf32>) -> tensor<64x256x512xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %indices_d0 = tensor.dim %indices, %c0 : tensor<64x256xi64>
    %indices_d1 = tensor.dim %indices, %c1 : tensor<64x256xi64>
    %table_d0 = tensor.dim %table, %c0 : tensor<20000x512xf32>
    %table_d1 = tensor.dim %table, %c1 : tensor<20000x512xf32>
    %out = tensor.empty() : tensor<64x256x512xf32>
    %t_em_d0 = scf.for %em_b0 = %c0 to %indices_d0 step %c1 iter_args(%t_em_em_b0 = %out) -> (tensor<64x256x512xf32>) {
    %t_em_d1 = scf.for %em_b1 = %c0 to %indices_d1 step %c1 iter_args(%t_em_em_b1 = %t_em_em_b0) -> (tensor<64x256x512xf32>) {
    %t_em_d2 = scf.for %em_d0 = %c0 to %table_d1 step %c1 iter_args(%t_em_em_d0 = %t_em_em_b1) -> (tensor<64x256x512xf32>) {
    %raw_idx = tensor.extract %indices[%em_b0, %em_b1] : tensor<64x256xi64>
    %row_idx = arith.index_cast %raw_idx : i64 to index
    %tv      = tensor.extract %table[%row_idx, %em_d0] : tensor<20000x512xf32>
    %t_ins   = tensor.insert %tv into %t_em_em_d0[%em_b0, %em_b1, %em_d0] : tensor<64x256x512xf32>
    scf.yield %t_ins : tensor<64x256x512xf32>
    }
    scf.yield %t_em_d2 : tensor<64x256x512xf32>
    }
    scf.yield %t_em_d1 : tensor<64x256x512xf32>
    }
    return %t_em_d0 : tensor<64x256x512xf32>
  }
}

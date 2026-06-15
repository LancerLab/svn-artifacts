module @f_19_vit_32x197_1000x768_32x197x768 {
  func.func @f_19_vit_32x197_1000x768_32x197x768(%table: memref<1000x768xf32>, %indices: memref<32x197xi64>) -> memref<32x197x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %indices_d0 = memref.dim %indices, %c0 : memref<32x197xi64>
    %indices_d1 = memref.dim %indices, %c1 : memref<32x197xi64>
    %table_d0 = memref.dim %table, %c0 : memref<1000x768xf32>
    %table_d1 = memref.dim %table, %c1 : memref<1000x768xf32>
    %out = memref.alloc() : memref<32x197x768xf32>
    scf.for %em_b0 = %c0 to %indices_d0 step %c1 {
      scf.for %em_b1 = %c0 to %indices_d1 step %c1 {
        scf.for %em_d0 = %c0 to %table_d1 step %c1 {
          %raw_idx = memref.load %indices[%em_b0, %em_b1] : memref<32x197xi64>
          %row_idx = arith.index_cast %raw_idx : i64 to index
          %tv      = memref.load %table[%row_idx, %em_d0] : memref<1000x768xf32>
          memref.store %tv, %out[%em_b0, %em_b1, %em_d0] : memref<32x197x768xf32>
        }
      }
    }
    return %out : memref<32x197x768xf32>
  }
}

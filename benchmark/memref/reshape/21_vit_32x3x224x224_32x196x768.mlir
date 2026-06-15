module @f_21_vit_32x3x224x224_32x196x768 {
  func.func @f_21_vit_32x3x224x224_32x196x768(%input: memref<32x3x224x224xf32>) -> memref<32x196x768xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %flat_in   = memref.collapse_shape %input [[0, 1, 2, 3]] : memref<32x3x224x224xf32> into memref<4816896xf32>
    %flat_out  = memref.alloc() : memref<4816896xf32>
    %flat_n    = arith.constant 4816896 : index
    scf.for %fi = %c0 to %flat_n step %c1 {
      %fv = memref.load %flat_in[%fi] : memref<4816896xf32>
      memref.store %fv, %flat_out[%fi] : memref<4816896xf32>
    }
    %out = memref.expand_shape %flat_out [[0, 1, 2]] output_shape [32, 196, 768] : memref<4816896xf32> into memref<32x196x768xf32>
    return %out : memref<32x196x768xf32>
  }
}

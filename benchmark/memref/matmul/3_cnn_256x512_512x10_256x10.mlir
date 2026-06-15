module @f_3_cnn_256x512_512x10_256x10 {
  func.func @f_3_cnn_256x512_512x10_256x10(%lhs: memref<256x512xf32>, %rhs: memref<512x10xf32>) -> memref<256x10xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lhs_d0 = memref.dim %lhs, %c0 : memref<256x512xf32>
    %lhs_d1 = memref.dim %lhs, %c1 : memref<256x512xf32>
    %rhs_d0 = memref.dim %rhs, %c0 : memref<512x10xf32>
    %rhs_d1 = memref.dim %rhs, %c1 : memref<512x10xf32>
    %out = memref.alloc() : memref<256x10xf32>
    %_meq0 = arith.cmpi eq, %lhs_d1, %rhs_d0 : index
    cf.assert %_meq0, "lhs.dim(1)==rhs.dim(0)"
    scf.for %m = %c0 to %lhs_d0 step %c1 {
      scf.for %n = %c0 to %rhs_d1 step %c1 {
        %zero_f = arith.constant 0.0 : f32
        %acc = scf.for %k = %c0 to %lhs_d1 step %c1 iter_args(%s = %zero_f) -> (f32) {
          %a  = memref.load %lhs[%m, %k] : memref<256x512xf32>
          %b  = memref.load %rhs[%k, %n] : memref<512x10xf32>
          %p  = arith.mulf %a, %b : f32
          %ns = arith.addf %s, %p : f32
          scf.yield %ns : f32
        }
        memref.store %acc, %out[%m, %n] : memref<256x10xf32>
      }
    }
    return %out : memref<256x10xf32>
  }
}

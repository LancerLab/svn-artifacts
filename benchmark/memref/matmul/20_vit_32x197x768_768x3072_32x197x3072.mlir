module @f_20_vit_32x197x768_768x3072_32x197x3072 {
  func.func @f_20_vit_32x197x768_768x3072_32x197x3072(%lhs: memref<32x197x768xf32>, %rhs: memref<768x3072xf32>) -> memref<32x197x3072xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %lhs_d0 = memref.dim %lhs, %c0 : memref<32x197x768xf32>
    %lhs_d1 = memref.dim %lhs, %c1 : memref<32x197x768xf32>
    %lhs_d2 = memref.dim %lhs, %c2 : memref<32x197x768xf32>
    %rhs_d0 = memref.dim %rhs, %c0 : memref<768x3072xf32>
    %rhs_d1 = memref.dim %rhs, %c1 : memref<768x3072xf32>
    %out = memref.alloc() : memref<32x197x3072xf32>
    %_meq0 = arith.cmpi eq, %lhs_d2, %rhs_d0 : index
    cf.assert %_meq0, "lhs.dim(2)==rhs.dim(0)"
    scf.for %bs = %c0 to %lhs_d0 step %c1 {
      scf.for %m = %c0 to %lhs_d1 step %c1 {
        scf.for %n = %c0 to %rhs_d1 step %c1 {
          %zero_f = arith.constant 0.0 : f32
          %acc = scf.for %k = %c0 to %lhs_d2 step %c1 iter_args(%s = %zero_f) -> (f32) {
            %a  = memref.load %lhs[%bs, %m, %k] : memref<32x197x768xf32>
            %b  = memref.load %rhs[%k, %n] : memref<768x3072xf32>
            %p  = arith.mulf %a, %b : f32
            %ns = arith.addf %s, %p : f32
            scf.yield %ns : f32
          }
          memref.store %acc, %out[%bs, %m, %n] : memref<32x197x3072xf32>
        }
      }
    }
    return %out : memref<32x197x3072xf32>
  }
}

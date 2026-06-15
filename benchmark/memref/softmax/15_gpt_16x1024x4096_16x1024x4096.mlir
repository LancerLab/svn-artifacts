module @f_15_gpt_16x1024x4096_16x1024x4096 {
  func.func @f_15_gpt_16x1024x4096_16x1024x4096(%input: memref<16x1024x4096xf32>) -> memref<16x1024x4096xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = memref.dim %input, %c0 : memref<16x1024x4096xf32>
    %input_d1 = memref.dim %input, %c1 : memref<16x1024x4096xf32>
    %input_d2 = memref.dim %input, %c2 : memref<16x1024x4096xf32>
    %out = memref.alloc() : memref<16x1024x4096xf32>
    scf.for %so0 = %c0 to %input_d0 step %c1 {
      scf.for %so1 = %c0 to %input_d1 step %c1 {
        %neg_inf = arith.constant -3.4028234663852886e+38 : f32
        %max_val = scf.for %sk = %c0 to %input_d2 step %c1 iter_args(%mx = %neg_inf) -> (f32) {
          %pv1 = memref.load %input[%so0, %so1, %sk] : memref<16x1024x4096xf32>
          %gt  = arith.cmpf ogt, %pv1, %mx : f32
          %nx  = arith.select %gt, %pv1, %mx : f32
          scf.yield %nx : f32
        }
        %zero_f  = arith.constant 0.0 : f32
        %sum_val = scf.for %sk2 = %c0 to %input_d2 step %c1 iter_args(%sm = %zero_f) -> (f32) {
          %pv2     = memref.load %input[%so0, %so1, %sk2] : memref<16x1024x4096xf32>
          %shifted = arith.subf %pv2, %max_val : f32
          %expv    = math.exp %shifted : f32
          %ns2     = arith.addf %sm, %expv : f32
          scf.yield %ns2 : f32
        }
        scf.for %sk3 = %c0 to %input_d2 step %c1 {
          %pv3  = memref.load %input[%so0, %so1, %sk3] : memref<16x1024x4096xf32>
          %sh3  = arith.subf %pv3, %max_val : f32
          %ex3  = math.exp %sh3 : f32
          %ov   = arith.divf %ex3, %sum_val : f32
          memref.store %ov, %out[%so0, %so1, %sk3] : memref<16x1024x4096xf32>
        }
      }
    }
    return %out : memref<16x1024x4096xf32>
  }
}

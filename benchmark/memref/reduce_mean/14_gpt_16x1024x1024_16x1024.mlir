module @f_14_gpt_16x1024x1024_16x1024 {
  func.func @f_14_gpt_16x1024x1024_16x1024(%input: memref<16x1024x1024xf32>) -> memref<16x1024xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = memref.dim %input, %c0 : memref<16x1024x1024xf32>
    %input_d1 = memref.dim %input, %c1 : memref<16x1024x1024xf32>
    %input_d2 = memref.dim %input, %c2 : memref<16x1024x1024xf32>
    %out = memref.alloc() : memref<16x1024xf32>
    %nsz = arith.constant 1024.0 : f32
    scf.for %rm_b0 = %c0 to %input_d0 step %c1 {
      scf.for %rm_b1 = %c0 to %input_d1 step %c1 {
        %zero_f = arith.constant 0.0 : f32
        %rm_sum = scf.for %rm_k = %c0 to %input_d2 step %c1 iter_args(%rm_s = %zero_f) -> (f32) {
          %rv   = memref.load %input[%rm_b0, %rm_b1, %rm_k] : memref<16x1024x1024xf32>
          %ns   = arith.addf %rm_s, %rv : f32
          scf.yield %ns : f32
        }
        %rm_mean = arith.divf %rm_sum, %nsz : f32
        memref.store %rm_mean, %out[%rm_b0, %rm_b1] : memref<16x1024xf32>
      }
    }
    return %out : memref<16x1024xf32>
  }
}

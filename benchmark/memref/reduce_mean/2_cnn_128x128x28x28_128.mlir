module @f_2_cnn_128x128x28x28_128 {
  func.func @f_2_cnn_128x128x28x28_128(%input: memref<128x128x28x28xf32>) -> memref<128xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = memref.dim %input, %c0 : memref<128x128x28x28xf32>
    %input_d1 = memref.dim %input, %c1 : memref<128x128x28x28xf32>
    %input_d2 = memref.dim %input, %c2 : memref<128x128x28x28xf32>
    %input_d3 = memref.dim %input, %c3 : memref<128x128x28x28xf32>
    %out = memref.alloc() : memref<128xf32>
    %nsz = arith.constant 100352.0 : f32
    scf.for %rm_b0 = %c0 to %input_d0 step %c1 {
      %zero_f = arith.constant 0.0 : f32
      %rm_sum = scf.for %rm_r0 = %c0 to %input_d1 step %c1 iter_args(%rm_s0 = %zero_f) -> (f32) {
        %rm_sum1 = scf.for %rm_r1 = %c0 to %input_d2 step %c1 iter_args(%rm_s1 = %rm_s0) -> (f32) {
        %rm_sum2 = scf.for %rm_r2 = %c0 to %input_d3 step %c1 iter_args(%rm_s2 = %rm_s1) -> (f32) {
        %rv   = memref.load %input[%rm_b0, %rm_r0, %rm_r1, %rm_r2] : memref<128x128x28x28xf32>
        %ns   = arith.addf %rm_s2, %rv : f32
        scf.yield %ns : f32
          }
        scf.yield %rm_sum2 : f32
        }
        scf.yield %rm_sum1 : f32
      }
      %rm_mean = arith.divf %rm_sum, %nsz : f32
      memref.store %rm_mean, %out[%rm_b0] : memref<128xf32>
    }
    return %out : memref<128xf32>
  }
}

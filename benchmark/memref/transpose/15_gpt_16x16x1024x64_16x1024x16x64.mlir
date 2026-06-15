module @f_15_gpt_16x16x1024x64_16x1024x16x64 {
  func.func @f_15_gpt_16x16x1024x64_16x1024x16x64(%input: memref<16x16x1024x64xf32>) -> memref<16x1024x16x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = memref.dim %input, %c0 : memref<16x16x1024x64xf32>
    %input_d1 = memref.dim %input, %c1 : memref<16x16x1024x64xf32>
    %input_d2 = memref.dim %input, %c2 : memref<16x16x1024x64xf32>
    %input_d3 = memref.dim %input, %c3 : memref<16x16x1024x64xf32>
    %out = memref.alloc() : memref<16x1024x16x64xf32>
    scf.for %pi0 = %c0 to %input_d0 step %c1 {
      scf.for %pi1 = %c0 to %input_d2 step %c1 {
        scf.for %pi2 = %c0 to %input_d1 step %c1 {
          scf.for %pi3 = %c0 to %input_d3 step %c1 {
            %tv    = memref.load %input[%pi0, %pi2, %pi1, %pi3] : memref<16x16x1024x64xf32>
            memref.store %tv, %out[%pi0, %pi1, %pi2, %pi3] : memref<16x1024x16x64xf32>
          }
        }
      }
    }
    return %out : memref<16x1024x16x64xf32>
  }
}

module @f_17_mobilenet_128x96x112x112 {
  func.func @f_17_mobilenet_128x96x112x112(%input: memref<128x96x112x112xf32>) -> memref<128x96x112x112xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = memref.dim %input, %c0 : memref<128x96x112x112xf32>
    %input_d1 = memref.dim %input, %c1 : memref<128x96x112x112xf32>
    %input_d2 = memref.dim %input, %c2 : memref<128x96x112x112xf32>
    %input_d3 = memref.dim %input, %c3 : memref<128x96x112x112xf32>
    %out = memref.alloc() : memref<128x96x112x112xf32>
    scf.for %gi0 = %c0 to %input_d0 step %c1 {
      scf.for %gi1 = %c0 to %input_d1 step %c1 {
        scf.for %gi2 = %c0 to %input_d2 step %c1 {
          scf.for %gi3 = %c0 to %input_d3 step %c1 {
            %xv      = memref.load %input[%gi0, %gi1, %gi2, %gi3] : memref<128x96x112x112xf32>
            %half    = arith.constant 5.000000e-01 : f32
            %one     = arith.constant 1.0 : f32
            %sqrt2pi = arith.constant 7.978846e-01 : f32
            %coeff   = arith.constant 4.471500e-02 : f32
            %x2      = arith.mulf %xv, %xv : f32
            %x3      = arith.mulf %x2, %xv : f32
            %cx3     = arith.mulf %coeff, %x3 : f32
            %inner   = arith.addf %xv, %cx3 : f32
            %targ    = arith.mulf %sqrt2pi, %inner : f32
            %tv      = math.tanh %targ : f32
            %one_tv  = arith.addf %one, %tv : f32
            %hx      = arith.mulf %half, %xv : f32
            %out_val = arith.mulf %hx, %one_tv : f32
            memref.store %out_val, %out[%gi0, %gi1, %gi2, %gi3] : memref<128x96x112x112xf32>
          }
        }
      }
    }
    return %out : memref<128x96x112x112xf32>
  }
}

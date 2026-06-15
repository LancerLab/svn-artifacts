module @f_19_vgg_32x512x14x14_32x512x7x7 {
  func.func @f_19_vgg_32x512x14x14_32x512x7x7(%input: memref<32x512x14x14xf32>) -> memref<32x512x7x7xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = memref.dim %input, %c0 : memref<32x512x14x14xf32>
    %input_d1 = memref.dim %input, %c1 : memref<32x512x14x14xf32>
    %input_d2 = memref.dim %input, %c2 : memref<32x512x14x14xf32>
    %input_d3 = memref.dim %input, %c3 : memref<32x512x14x14xf32>
    %out = memref.alloc() : memref<32x512x7x7xf32>
    %kH_sz  = arith.constant 2 : index
    %kW_sz  = arith.constant 2 : index
    %stride = arith.constant 2 : index
    %out_d2 = arith.constant 7 : index
    %out_d3 = arith.constant 7 : index
    scf.for %n = %c0 to %input_d0 step %c1 {
      scf.for %c = %c0 to %input_d1 step %c1 {
        scf.for %oh = %c0 to %out_d2 step %c1 {
          scf.for %ow = %c0 to %out_d3 step %c1 {
            %neg_inf = arith.constant -3.4028234663852886e+38 : f32
            %max_v = scf.for %kh = %c0 to %kH_sz step %c1 iter_args(%mx_h = %neg_inf) -> (f32) {
              %max_v2 = scf.for %kw = %c0 to %kW_sz step %c1 iter_args(%mx_w = %mx_h) -> (f32) {
                %s_oh = arith.muli %stride, %oh : index
                %ih   = arith.addi %s_oh, %kh : index
                %s_ow = arith.muli %stride, %ow : index
                %iw   = arith.addi %s_ow, %kw : index
                %pv   = memref.load %input[%n, %c, %ih, %iw] : memref<32x512x14x14xf32>
                %gt   = arith.cmpf ogt, %pv, %mx_w : f32
                %nx   = arith.select %gt, %pv, %mx_w : f32
                scf.yield %nx : f32
              }
              scf.yield %max_v2 : f32
            }
            memref.store %max_v, %out[%n, %c, %oh, %ow] : memref<32x512x7x7xf32>
          }
        }
      }
    }
    return %out : memref<32x512x7x7xf32>
  }
}

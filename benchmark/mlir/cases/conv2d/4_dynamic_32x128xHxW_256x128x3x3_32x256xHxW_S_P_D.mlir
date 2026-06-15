module {
  func.func @f_4_dynamic_32x128xHxW_256x128x3x3_32x256xHxW_S_P_D(%input: tensor<32x128x?x?xf32>, %filter: tensor<256x128x3x3xf32>) -> tensor<32x256x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %zero_f = arith.constant 0.0 : f32
    %input_d0 = tensor.dim %input, %c0 : tensor<32x128x?x?xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<32x128x?x?xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<32x128x?x?xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<32x128x?x?xf32>
    %filter_d0 = tensor.dim %filter, %c0 : tensor<256x128x3x3xf32>
    %filter_d1 = tensor.dim %filter, %c1 : tensor<256x128x3x3xf32>
    %filter_d2 = tensor.dim %filter, %c2 : tensor<256x128x3x3xf32>
    %filter_d3 = tensor.dim %filter, %c3 : tensor<256x128x3x3xf32>
    %eq0 = arith.cmpi eq, %input_d1, %filter_d1 : index
    cf.assert %eq0, "input.dim(1)==filter.dim(1)"
    %out = tensor.empty(%input_d2, %input_d3) : tensor<32x256x?x?xf32>
    %stride = arith.constant 1 : index
    %kH_sz  = arith.constant 3 : index
    %kW_sz  = arith.constant 3 : index
    %t_res_n = scf.for %n = %c0 to %input_d0 step %c1 iter_args(%t_cv_n = %out) -> (tensor<32x256x?x?xf32>) {
    %t_res_f = scf.for %f = %c0 to %filter_d0 step %c1 iter_args(%t_cv_f = %t_cv_n) -> (tensor<32x256x?x?xf32>) {
    %t_res_oh = scf.for %oh = %c0 to %input_d2 step %c1 iter_args(%t_cv_oh = %t_cv_f) -> (tensor<32x256x?x?xf32>) {
    %t_res_ow = scf.for %ow = %c0 to %input_d3 step %c1 iter_args(%t_cv_ow = %t_cv_oh) -> (tensor<32x256x?x?xf32>) {
    %acc = scf.for %c = %c0 to %input_d1 step %c1 iter_args(%s_c = %zero_f) -> (f32) {
    %acc2 = scf.for %kh = %c0 to %kH_sz step %c1 iter_args(%s_kh = %s_c) -> (f32) {
    %acc3 = scf.for %kw = %c0 to %kW_sz step %c1 iter_args(%s_kw = %s_kh) -> (f32) {
    %s_oh = arith.muli %stride, %oh : index
    %ih   = arith.addi %s_oh, %kh : index
    %s_ow = arith.muli %stride, %ow : index
    %iw   = arith.addi %s_ow, %kw : index
    %xv   = tensor.extract %input[%n, %c, %ih, %iw]  : tensor<32x128x?x?xf32>
    %wv   = tensor.extract %filter[%f, %c, %kh, %kw] : tensor<256x128x3x3xf32>
    %ml   = arith.mulf %xv, %wv : f32
    %sm   = arith.addf %s_kw, %ml : f32
    scf.yield %sm : f32
    }
    scf.yield %acc3 : f32
    }
    scf.yield %acc2 : f32
    }
    %t_new = tensor.insert %acc into %t_cv_ow[%n, %f, %oh, %ow] : tensor<32x256x?x?xf32>
    scf.yield %t_new : tensor<32x256x?x?xf32>
    }
    scf.yield %t_res_ow : tensor<32x256x?x?xf32>
    }
    scf.yield %t_res_oh : tensor<32x256x?x?xf32>
    }
    scf.yield %t_res_f : tensor<32x256x?x?xf32>
    }
    return %t_res_n : tensor<32x256x?x?xf32>
  }
}

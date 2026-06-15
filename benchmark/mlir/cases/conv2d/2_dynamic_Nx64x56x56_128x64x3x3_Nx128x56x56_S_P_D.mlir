module {
  func.func @f_2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D(%input: tensor<?x64x56x56xf32>, %filter: tensor<128x64x3x3xf32>) -> tensor<?x128x56x56xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %zero_f = arith.constant 0.0 : f32
    %input_d0 = tensor.dim %input, %c0 : tensor<?x64x56x56xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<?x64x56x56xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<?x64x56x56xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<?x64x56x56xf32>
    %filter_d0 = tensor.dim %filter, %c0 : tensor<128x64x3x3xf32>
    %filter_d1 = tensor.dim %filter, %c1 : tensor<128x64x3x3xf32>
    %filter_d2 = tensor.dim %filter, %c2 : tensor<128x64x3x3xf32>
    %filter_d3 = tensor.dim %filter, %c3 : tensor<128x64x3x3xf32>
    %eq0 = arith.cmpi eq, %input_d1, %filter_d1 : index
    cf.assert %eq0, "input.dim(1)==filter.dim(1)"
    %out = tensor.empty(%input_d0) : tensor<?x128x56x56xf32>
    %stride = arith.constant 1 : index
    %kH_sz  = arith.constant 3 : index
    %kW_sz  = arith.constant 3 : index
    %out_sz2 = arith.constant 56 : index
    %out_sz3 = arith.constant 56 : index
    %t_res_n = scf.for %n = %c0 to %input_d0 step %c1 iter_args(%t_cv_n = %out) -> (tensor<?x128x56x56xf32>) {
    %t_res_f = scf.for %f = %c0 to %filter_d0 step %c1 iter_args(%t_cv_f = %t_cv_n) -> (tensor<?x128x56x56xf32>) {
    %t_res_oh = scf.for %oh = %c0 to %out_sz2 step %c1 iter_args(%t_cv_oh = %t_cv_f) -> (tensor<?x128x56x56xf32>) {
    %t_res_ow = scf.for %ow = %c0 to %out_sz3 step %c1 iter_args(%t_cv_ow = %t_cv_oh) -> (tensor<?x128x56x56xf32>) {
    %acc = scf.for %c = %c0 to %input_d1 step %c1 iter_args(%s_c = %zero_f) -> (f32) {
    %acc2 = scf.for %kh = %c0 to %kH_sz step %c1 iter_args(%s_kh = %s_c) -> (f32) {
    %acc3 = scf.for %kw = %c0 to %kW_sz step %c1 iter_args(%s_kw = %s_kh) -> (f32) {
    %s_oh = arith.muli %stride, %oh : index
    %ih   = arith.addi %s_oh, %kh : index
    %s_ow = arith.muli %stride, %ow : index
    %iw   = arith.addi %s_ow, %kw : index
    %xv   = tensor.extract %input[%n, %c, %ih, %iw]  : tensor<?x64x56x56xf32>
    %wv   = tensor.extract %filter[%f, %c, %kh, %kw] : tensor<128x64x3x3xf32>
    %ml   = arith.mulf %xv, %wv : f32
    %sm   = arith.addf %s_kw, %ml : f32
    scf.yield %sm : f32
    }
    scf.yield %acc3 : f32
    }
    scf.yield %acc2 : f32
    }
    %t_new = tensor.insert %acc into %t_cv_ow[%n, %f, %oh, %ow] : tensor<?x128x56x56xf32>
    scf.yield %t_new : tensor<?x128x56x56xf32>
    }
    scf.yield %t_res_ow : tensor<?x128x56x56xf32>
    }
    scf.yield %t_res_oh : tensor<?x128x56x56xf32>
    }
    scf.yield %t_res_f : tensor<?x128x56x56xf32>
    }
    return %t_res_n : tensor<?x128x56x56xf32>
  }
}

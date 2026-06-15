module {
  func.func @f_13_dynamic_BxCx128x128_BxCx32x32(%input: tensor<?x?x128x128xf32>) -> tensor<?x?x32x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %neg_inf = arith.constant -3.4028234663852886e+38 : f32
    %input_d0 = tensor.dim %input, %c0 : tensor<?x?x128x128xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<?x?x128x128xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<?x?x128x128xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<?x?x128x128xf32>
    %out = tensor.empty(%input_d0, %input_d1) : tensor<?x?x32x32xf32>
    %kh_sz  = arith.constant 4 : index
    %kw_sz  = arith.constant 4 : index
    %stride = arith.constant 4 : index
    %po_sz2 = arith.constant 32 : index
    %po_sz3 = arith.constant 32 : index
    %t_res_pn = scf.for %pn = %c0 to %input_d0 step %c1 iter_args(%t_mp_pn = %out) -> (tensor<?x?x32x32xf32>) {
    %t_res_pc = scf.for %pc = %c0 to %input_d1 step %c1 iter_args(%t_mp_pc = %t_mp_pn) -> (tensor<?x?x32x32xf32>) {
    %t_res_poh = scf.for %poh = %c0 to %po_sz2 step %c1 iter_args(%t_mp_poh = %t_mp_pc) -> (tensor<?x?x32x32xf32>) {
    %t_res_pow = scf.for %pow = %c0 to %po_sz3 step %c1 iter_args(%t_mp_pow = %t_mp_poh) -> (tensor<?x?x32x32xf32>) {
    %pool = scf.for %pkh = %c0 to %kh_sz step %c1 iter_args(%mp_kh = %neg_inf) -> (f32) {
    %pool2 = scf.for %pkw = %c0 to %kw_sz step %c1 iter_args(%mp_kw = %mp_kh) -> (f32) {
    %s_poh = arith.muli %stride, %poh : index
    %ih    = arith.addi %s_poh, %pkh : index
    %s_pow = arith.muli %stride, %pow : index
    %iw    = arith.addi %s_pow, %pkw : index
    %pv    = tensor.extract %input[%pn, %pc, %ih, %iw] : tensor<?x?x128x128xf32>
    %gt    = arith.cmpf ogt, %pv, %mp_kw : f32
    %mx    = arith.select %gt, %pv, %mp_kw : f32
    scf.yield %mx : f32
    }
    scf.yield %pool2 : f32
    }
    %t_new = tensor.insert %pool into %t_mp_pow[%pn, %pc, %poh, %pow] : tensor<?x?x32x32xf32>
    scf.yield %t_new : tensor<?x?x32x32xf32>
    }
    scf.yield %t_res_pow : tensor<?x?x32x32xf32>
    }
    scf.yield %t_res_poh : tensor<?x?x32x32xf32>
    }
    scf.yield %t_res_pc : tensor<?x?x32x32xf32>
    }
    return %t_res_pn : tensor<?x?x32x32xf32>
  }
}

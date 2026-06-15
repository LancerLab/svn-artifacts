module {
  func.func @f_4_dynamic_Bx256x56x56_Bx256x56x56(%input: tensor<?x256x56x56xf32>) -> tensor<?x256x56x56xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<?x256x56x56xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<?x256x56x56xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<?x256x56x56xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<?x256x56x56xf32>
    %out = tensor.empty(%input_d0) : tensor<?x256x56x56xf32>
    %t_res_so0 = scf.for %so0 = %c0 to %input_d0 step %c1 iter_args(%t_sm_so0 = %out) -> (tensor<?x256x56x56xf32>) {
    %t_res_so1 = scf.for %so1 = %c0 to %input_d1 step %c1 iter_args(%t_sm_so1 = %t_sm_so0) -> (tensor<?x256x56x56xf32>) {
    %t_res_so2 = scf.for %so2 = %c0 to %input_d2 step %c1 iter_args(%t_sm_so2 = %t_sm_so1) -> (tensor<?x256x56x56xf32>) {
    %neg_inf = arith.constant -3.4028234663852886e+38 : f32
    %max_val = scf.for %sk = %c0 to %input_d3 step %c1 iter_args(%mx = %neg_inf) -> (f32) {
    %pv1 = tensor.extract %input[%so0, %so1, %so2, %sk] : tensor<?x256x56x56xf32>
    %gt  = arith.cmpf ogt, %pv1, %mx : f32
    %nx  = arith.select %gt, %pv1, %mx : f32
    scf.yield %nx : f32
    }
    %zero_f  = arith.constant 0.0 : f32
    %sum_val = scf.for %sk = %c0 to %input_d3 step %c1 iter_args(%sm = %zero_f) -> (f32) {
    %pv2     = tensor.extract %input[%so0, %so1, %so2, %sk] : tensor<?x256x56x56xf32>
    %shifted = arith.subf %pv2, %max_val : f32
    %expv    = math.exp %shifted : f32
    %ns2     = arith.addf %sm, %expv : f32
    scf.yield %ns2 : f32
    }
    %t_res_sk = scf.for %sk = %c0 to %input_d3 step %c1 iter_args(%t_sm_sk = %t_sm_so2) -> (tensor<?x256x56x56xf32>) {
    %pv3  = tensor.extract %input[%so0, %so1, %so2, %sk] : tensor<?x256x56x56xf32>
    %sh3  = arith.subf %pv3, %max_val : f32
    %ex3  = math.exp %sh3 : f32
    %norm = arith.divf %ex3, %sum_val : f32
    %t_sk_ins = tensor.insert %norm into %t_sm_sk[%so0, %so1, %so2, %sk] : tensor<?x256x56x56xf32>
    scf.yield %t_sk_ins : tensor<?x256x56x56xf32>
    }
    scf.yield %t_res_sk : tensor<?x256x56x56xf32>
    }
    scf.yield %t_res_so2 : tensor<?x256x56x56xf32>
    }
    scf.yield %t_res_so1 : tensor<?x256x56x56xf32>
    }
    return %t_res_so0 : tensor<?x256x56x56xf32>
  }
}

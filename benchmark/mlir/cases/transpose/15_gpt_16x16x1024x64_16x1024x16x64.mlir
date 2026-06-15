module {
  func.func @f_15_gpt_16x16x1024x64_16x1024x16x64(%input: tensor<16x16x1024x64xf32>) -> tensor<16x1024x16x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<16x16x1024x64xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<16x16x1024x64xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<16x16x1024x64xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<16x16x1024x64xf32>
    %out = tensor.empty() : tensor<16x1024x16x64xf32>
    %t_res_pi0 = scf.for %pi0 = %c0 to %input_d0 step %c1 iter_args(%t_tr_pi0 = %out) -> (tensor<16x1024x16x64xf32>) {
    %t_res_pi1 = scf.for %pi1 = %c0 to %input_d2 step %c1 iter_args(%t_tr_pi1 = %t_tr_pi0) -> (tensor<16x1024x16x64xf32>) {
    %t_res_pi2 = scf.for %pi2 = %c0 to %input_d1 step %c1 iter_args(%t_tr_pi2 = %t_tr_pi1) -> (tensor<16x1024x16x64xf32>) {
    %t_res_pi3 = scf.for %pi3 = %c0 to %input_d3 step %c1 iter_args(%t_tr_pi3 = %t_tr_pi2) -> (tensor<16x1024x16x64xf32>) {
    %tv = tensor.extract %input[%pi0, %pi2, %pi1, %pi3] : tensor<16x16x1024x64xf32>
    %t_ins = tensor.insert %tv into %t_tr_pi3[%pi0, %pi1, %pi2, %pi3] : tensor<16x1024x16x64xf32>
    scf.yield %t_ins : tensor<16x1024x16x64xf32>
    }
    scf.yield %t_res_pi3 : tensor<16x1024x16x64xf32>
    }
    scf.yield %t_res_pi2 : tensor<16x1024x16x64xf32>
    }
    scf.yield %t_res_pi1 : tensor<16x1024x16x64xf32>
    }
    return %t_res_pi0 : tensor<16x1024x16x64xf32>
  }
}

module {
  func.func @f_16_lstm_64x100x256_256_256_64x100x256(%input: tensor<64x100x256xf32>, %gamma: tensor<256xf32>, %beta: tensor<256xf32>) -> tensor<64x100x256xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<64x100x256xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<64x100x256xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<64x100x256xf32>
    %gamma_d0 = tensor.dim %gamma, %c0 : tensor<256xf32>
    %beta_d0 = tensor.dim %beta, %c0 : tensor<256xf32>
    %eq0 = arith.cmpi eq, %gamma_d0, %input_d2 : index
    cf.assert %eq0, "gamma.dim(0)==input.dim(2)"
    %eq1 = arith.cmpi eq, %beta_d0, %input_d2 : index
    cf.assert %eq1, "beta.dim(0)==input.dim(2)"
    %out = tensor.empty() : tensor<64x100x256xf32>
    %t_res_bn0 = scf.for %bn0 = %c0 to %input_d0 step %c1 iter_args(%t_bn_bn0 = %out) -> (tensor<64x100x256xf32>) {
    %t_res_bn1 = scf.for %bn1 = %c0 to %input_d1 step %c1 iter_args(%t_bn_bn1 = %t_bn_bn0) -> (tensor<64x100x256xf32>) {
    %t_res_bn2 = scf.for %bn2 = %c0 to %input_d2 step %c1 iter_args(%t_bn_bn2 = %t_bn_bn1) -> (tensor<64x100x256xf32>) {
    %xv  = tensor.extract %input[%bn0, %bn1, %bn2] : tensor<64x100x256xf32>
    %gv  = tensor.extract %gamma[%bn2] : tensor<256xf32>
    %bv  = tensor.extract %beta[%bn2]  : tensor<256xf32>
    %scl = arith.mulf %xv, %gv : f32
    %res = arith.addf %scl, %bv : f32
    %t_ins = tensor.insert %res into %t_bn_bn2[%bn0, %bn1, %bn2] : tensor<64x100x256xf32>
    scf.yield %t_ins : tensor<64x100x256xf32>
    }
    scf.yield %t_res_bn2 : tensor<64x100x256xf32>
    }
    scf.yield %t_res_bn1 : tensor<64x100x256xf32>
    }
    return %t_res_bn0 : tensor<64x100x256xf32>
  }
}

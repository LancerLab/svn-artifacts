module {
  func.func @f_21_vit_32x197x3072(%input: tensor<32x197x3072xf32>) -> tensor<32x197x3072xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<32x197x3072xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<32x197x3072xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<32x197x3072xf32>
    %out = tensor.empty() : tensor<32x197x3072xf32>
    %t_res_ui0 = scf.for %ui0 = %c0 to %input_d0 step %c1 iter_args(%t_un_ui0 = %out) -> (tensor<32x197x3072xf32>) {
    %t_res_ui1 = scf.for %ui1 = %c0 to %input_d1 step %c1 iter_args(%t_un_ui1 = %t_un_ui0) -> (tensor<32x197x3072xf32>) {
    %t_res_ui2 = scf.for %ui2 = %c0 to %input_d2 step %c1 iter_args(%t_un_ui2 = %t_un_ui1) -> (tensor<32x197x3072xf32>) {
    %in_val = tensor.extract %input[%ui0, %ui1, %ui2] : tensor<32x197x3072xf32>
    %neg = arith.negf %in_val : f32
    %exp = math.exp %neg : f32
    %one = arith.constant 1.0 : f32
    %den = arith.addf %one, %exp : f32
    %out_val = arith.divf %one, %den : f32
    %t_ins = tensor.insert %out_val into %t_un_ui2[%ui0, %ui1, %ui2] : tensor<32x197x3072xf32>
    scf.yield %t_ins : tensor<32x197x3072xf32>
    }
    scf.yield %t_res_ui2 : tensor<32x197x3072xf32>
    }
    scf.yield %t_res_ui1 : tensor<32x197x3072xf32>
    }
    return %t_res_ui0 : tensor<32x197x3072xf32>
  }
}

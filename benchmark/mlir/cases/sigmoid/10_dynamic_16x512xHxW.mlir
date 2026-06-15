module {
  func.func @f_10_dynamic_16x512xHxW(%input: tensor<16x512x?x?xf32>) -> tensor<16x512x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<16x512x?x?xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<16x512x?x?xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<16x512x?x?xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<16x512x?x?xf32>
    %out = tensor.empty(%input_d2, %input_d3) : tensor<16x512x?x?xf32>
    %t_res_ui0 = scf.for %ui0 = %c0 to %input_d0 step %c1 iter_args(%t_un_ui0 = %out) -> (tensor<16x512x?x?xf32>) {
    %t_res_ui1 = scf.for %ui1 = %c0 to %input_d1 step %c1 iter_args(%t_un_ui1 = %t_un_ui0) -> (tensor<16x512x?x?xf32>) {
    %t_res_ui2 = scf.for %ui2 = %c0 to %input_d2 step %c1 iter_args(%t_un_ui2 = %t_un_ui1) -> (tensor<16x512x?x?xf32>) {
    %t_res_ui3 = scf.for %ui3 = %c0 to %input_d3 step %c1 iter_args(%t_un_ui3 = %t_un_ui2) -> (tensor<16x512x?x?xf32>) {
    %in_val = tensor.extract %input[%ui0, %ui1, %ui2, %ui3] : tensor<16x512x?x?xf32>
    %neg = arith.negf %in_val : f32
    %exp = math.exp %neg : f32
    %one = arith.constant 1.0 : f32
    %den = arith.addf %one, %exp : f32
    %out_val = arith.divf %one, %den : f32
    %t_ins = tensor.insert %out_val into %t_un_ui3[%ui0, %ui1, %ui2, %ui3] : tensor<16x512x?x?xf32>
    scf.yield %t_ins : tensor<16x512x?x?xf32>
    }
    scf.yield %t_res_ui3 : tensor<16x512x?x?xf32>
    }
    scf.yield %t_res_ui2 : tensor<16x512x?x?xf32>
    }
    scf.yield %t_res_ui1 : tensor<16x512x?x?xf32>
    }
    return %t_res_ui0 : tensor<16x512x?x?xf32>
  }
}

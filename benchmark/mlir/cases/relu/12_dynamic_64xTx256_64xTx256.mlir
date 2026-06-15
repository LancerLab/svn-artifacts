module {
  func.func @f_12_dynamic_64xTx256_64xTx256(%input: tensor<64x?x256xf32>) -> tensor<64x?x256xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<64x?x256xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<64x?x256xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<64x?x256xf32>
    %out = tensor.empty(%input_d1) : tensor<64x?x256xf32>
    %t_res_ui0 = scf.for %ui0 = %c0 to %input_d0 step %c1 iter_args(%t_un_ui0 = %out) -> (tensor<64x?x256xf32>) {
    %t_res_ui1 = scf.for %ui1 = %c0 to %input_d1 step %c1 iter_args(%t_un_ui1 = %t_un_ui0) -> (tensor<64x?x256xf32>) {
    %t_res_ui2 = scf.for %ui2 = %c0 to %input_d2 step %c1 iter_args(%t_un_ui2 = %t_un_ui1) -> (tensor<64x?x256xf32>) {
    %in_val = tensor.extract %input[%ui0, %ui1, %ui2] : tensor<64x?x256xf32>
    %zf = arith.constant 0.0 : f32
    %out_val = arith.maximumf %in_val, %zf : f32
    %t_ins = tensor.insert %out_val into %t_un_ui2[%ui0, %ui1, %ui2] : tensor<64x?x256xf32>
    scf.yield %t_ins : tensor<64x?x256xf32>
    }
    scf.yield %t_res_ui2 : tensor<64x?x256xf32>
    }
    scf.yield %t_res_ui1 : tensor<64x?x256xf32>
    }
    return %t_res_ui0 : tensor<64x?x256xf32>
  }
}

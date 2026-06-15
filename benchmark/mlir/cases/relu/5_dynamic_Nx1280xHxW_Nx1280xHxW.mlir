module {
  func.func @f_5_dynamic_Nx1280xHxW_Nx1280xHxW(%input: tensor<?x1280x?x?xf32>) -> tensor<?x1280x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %input_d0 = tensor.dim %input, %c0 : tensor<?x1280x?x?xf32>
    %input_d1 = tensor.dim %input, %c1 : tensor<?x1280x?x?xf32>
    %input_d2 = tensor.dim %input, %c2 : tensor<?x1280x?x?xf32>
    %input_d3 = tensor.dim %input, %c3 : tensor<?x1280x?x?xf32>
    %out = tensor.empty(%input_d0, %input_d2, %input_d3) : tensor<?x1280x?x?xf32>
    %t_res_ui0 = scf.for %ui0 = %c0 to %input_d0 step %c1 iter_args(%t_un_ui0 = %out) -> (tensor<?x1280x?x?xf32>) {
    %t_res_ui1 = scf.for %ui1 = %c0 to %input_d1 step %c1 iter_args(%t_un_ui1 = %t_un_ui0) -> (tensor<?x1280x?x?xf32>) {
    %t_res_ui2 = scf.for %ui2 = %c0 to %input_d2 step %c1 iter_args(%t_un_ui2 = %t_un_ui1) -> (tensor<?x1280x?x?xf32>) {
    %t_res_ui3 = scf.for %ui3 = %c0 to %input_d3 step %c1 iter_args(%t_un_ui3 = %t_un_ui2) -> (tensor<?x1280x?x?xf32>) {
    %in_val = tensor.extract %input[%ui0, %ui1, %ui2, %ui3] : tensor<?x1280x?x?xf32>
    %zf = arith.constant 0.0 : f32
    %out_val = arith.maximumf %in_val, %zf : f32
    %t_ins = tensor.insert %out_val into %t_un_ui3[%ui0, %ui1, %ui2, %ui3] : tensor<?x1280x?x?xf32>
    scf.yield %t_ins : tensor<?x1280x?x?xf32>
    }
    scf.yield %t_res_ui3 : tensor<?x1280x?x?xf32>
    }
    scf.yield %t_res_ui2 : tensor<?x1280x?x?xf32>
    }
    scf.yield %t_res_ui1 : tensor<?x1280x?x?xf32>
    }
    return %t_res_ui0 : tensor<?x1280x?x?xf32>
  }
}

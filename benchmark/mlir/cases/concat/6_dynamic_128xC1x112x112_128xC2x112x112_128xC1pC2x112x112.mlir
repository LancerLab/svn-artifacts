module {
  func.func @f_6_dynamic_128xC1x112x112_128xC2x112x112_128xC1pC2x112x112(%in0: tensor<128x?x112x112xf32>, %in1: tensor<128x?x112x112xf32>) -> tensor<128x?x112x112xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %in0_d0 = tensor.dim %in0, %c0 : tensor<128x?x112x112xf32>
    %in0_d1 = tensor.dim %in0, %c1 : tensor<128x?x112x112xf32>
    %in0_d2 = tensor.dim %in0, %c2 : tensor<128x?x112x112xf32>
    %in0_d3 = tensor.dim %in0, %c3 : tensor<128x?x112x112xf32>
    %in1_d0 = tensor.dim %in1, %c0 : tensor<128x?x112x112xf32>
    %in1_d1 = tensor.dim %in1, %c1 : tensor<128x?x112x112xf32>
    %in1_d2 = tensor.dim %in1, %c2 : tensor<128x?x112x112xf32>
    %in1_d3 = tensor.dim %in1, %c3 : tensor<128x?x112x112xf32>
    %eq0 = arith.cmpi eq, %in0_d0, %in1_d0 : index
    cf.assert %eq0, "in0.dim(0)==in1.dim(0)"
    %eq1 = arith.cmpi eq, %in0_d2, %in1_d2 : index
    cf.assert %eq1, "in0.dim(2)==in1.dim(2)"
    %eq2 = arith.cmpi eq, %in0_d3, %in1_d3 : index
    cf.assert %eq2, "in0.dim(3)==in1.dim(3)"
    %csum0 = arith.addi %c0, %in0_d1 : index
    %csum1 = arith.addi %csum0, %in1_d1 : index
    %out = tensor.empty(%csum1) : tensor<128x?x112x112xf32>
    %t_cc0_d0 = scf.for %cc0_0 = %c0 to %in0_d0 step %c1 iter_args(%t_cc0_0 = %out) -> (tensor<128x?x112x112xf32>) {
    %t_cc0_d1 = scf.for %cc0_1 = %c0 to %in0_d1 step %c1 iter_args(%t_cc0_1 = %t_cc0_0) -> (tensor<128x?x112x112xf32>) {
    %t_cc0_d2 = scf.for %cc0_2 = %c0 to %in0_d2 step %c1 iter_args(%t_cc0_2 = %t_cc0_1) -> (tensor<128x?x112x112xf32>) {
    %t_cc0_d3 = scf.for %cc0_3 = %c0 to %in0_d3 step %c1 iter_args(%t_cc0_3 = %t_cc0_2) -> (tensor<128x?x112x112xf32>) {
    %cv0 = tensor.extract %in0[%cc0_0, %cc0_1, %cc0_2, %cc0_3] : tensor<128x?x112x112xf32>
    %t_ins_cc0 = tensor.insert %cv0 into %t_cc0_3[%cc0_0, %cc0_1, %cc0_2, %cc0_3] : tensor<128x?x112x112xf32>
    scf.yield %t_ins_cc0 : tensor<128x?x112x112xf32>
    }
    scf.yield %t_cc0_d3 : tensor<128x?x112x112xf32>
    }
    scf.yield %t_cc0_d2 : tensor<128x?x112x112xf32>
    }
    scf.yield %t_cc0_d1 : tensor<128x?x112x112xf32>
    }
    %t_cc1_d0 = scf.for %cc1_0 = %c0 to %in1_d0 step %c1 iter_args(%t_cc1_0 = %t_cc0_d0) -> (tensor<128x?x112x112xf32>) {
    %t_cc1_d1 = scf.for %cc1_1 = %c0 to %in1_d1 step %c1 iter_args(%t_cc1_1 = %t_cc1_0) -> (tensor<128x?x112x112xf32>) {
    %t_cc1_d2 = scf.for %cc1_2 = %c0 to %in1_d2 step %c1 iter_args(%t_cc1_2 = %t_cc1_1) -> (tensor<128x?x112x112xf32>) {
    %t_cc1_d3 = scf.for %cc1_3 = %c0 to %in1_d3 step %c1 iter_args(%t_cc1_3 = %t_cc1_2) -> (tensor<128x?x112x112xf32>) {
    %t_ax_off1 = arith.addi %in0_d1, %cc1_1 : index
    %cv1 = tensor.extract %in1[%cc1_0, %cc1_1, %cc1_2, %cc1_3] : tensor<128x?x112x112xf32>
    %t_ins_cc1 = tensor.insert %cv1 into %t_cc1_3[%cc1_0, %t_ax_off1, %cc1_2, %cc1_3] : tensor<128x?x112x112xf32>
    scf.yield %t_ins_cc1 : tensor<128x?x112x112xf32>
    }
    scf.yield %t_cc1_d3 : tensor<128x?x112x112xf32>
    }
    scf.yield %t_cc1_d2 : tensor<128x?x112x112xf32>
    }
    scf.yield %t_cc1_d1 : tensor<128x?x112x112xf32>
    }
    return %t_cc1_d0 : tensor<128x?x112x112xf32>
  }
}

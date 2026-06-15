module {
  func.func @f_19_transformer_32x512x64_32x512x64_32x512x64_32x512x64_32x512x256(%in0: tensor<32x512x64xf32>, %in1: tensor<32x512x64xf32>, %in2: tensor<32x512x64xf32>, %in3: tensor<32x512x64xf32>) -> tensor<32x512x256xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %in0_d0 = tensor.dim %in0, %c0 : tensor<32x512x64xf32>
    %in0_d1 = tensor.dim %in0, %c1 : tensor<32x512x64xf32>
    %in0_d2 = tensor.dim %in0, %c2 : tensor<32x512x64xf32>
    %in1_d0 = tensor.dim %in1, %c0 : tensor<32x512x64xf32>
    %in1_d1 = tensor.dim %in1, %c1 : tensor<32x512x64xf32>
    %in1_d2 = tensor.dim %in1, %c2 : tensor<32x512x64xf32>
    %in2_d0 = tensor.dim %in2, %c0 : tensor<32x512x64xf32>
    %in2_d1 = tensor.dim %in2, %c1 : tensor<32x512x64xf32>
    %in2_d2 = tensor.dim %in2, %c2 : tensor<32x512x64xf32>
    %in3_d0 = tensor.dim %in3, %c0 : tensor<32x512x64xf32>
    %in3_d1 = tensor.dim %in3, %c1 : tensor<32x512x64xf32>
    %in3_d2 = tensor.dim %in3, %c2 : tensor<32x512x64xf32>
    %eq0 = arith.cmpi eq, %in0_d0, %in1_d0 : index
    cf.assert %eq0, "in0.dim(0)==in1.dim(0)"
    %eq1 = arith.cmpi eq, %in0_d1, %in1_d1 : index
    cf.assert %eq1, "in0.dim(1)==in1.dim(1)"
    %eq2 = arith.cmpi eq, %in0_d0, %in2_d0 : index
    cf.assert %eq2, "in0.dim(0)==in2.dim(0)"
    %eq3 = arith.cmpi eq, %in0_d1, %in2_d1 : index
    cf.assert %eq3, "in0.dim(1)==in2.dim(1)"
    %eq4 = arith.cmpi eq, %in0_d0, %in3_d0 : index
    cf.assert %eq4, "in0.dim(0)==in3.dim(0)"
    %eq5 = arith.cmpi eq, %in0_d1, %in3_d1 : index
    cf.assert %eq5, "in0.dim(1)==in3.dim(1)"
    %out = tensor.empty() : tensor<32x512x256xf32>
    %t_cc0_d0 = scf.for %cc0_0 = %c0 to %in0_d0 step %c1 iter_args(%t_cc0_0 = %out) -> (tensor<32x512x256xf32>) {
    %t_cc0_d1 = scf.for %cc0_1 = %c0 to %in0_d1 step %c1 iter_args(%t_cc0_1 = %t_cc0_0) -> (tensor<32x512x256xf32>) {
    %t_cc0_d2 = scf.for %cc0_2 = %c0 to %in0_d2 step %c1 iter_args(%t_cc0_2 = %t_cc0_1) -> (tensor<32x512x256xf32>) {
    %cv0 = tensor.extract %in0[%cc0_0, %cc0_1, %cc0_2] : tensor<32x512x64xf32>
    %t_ins_cc0 = tensor.insert %cv0 into %t_cc0_2[%cc0_0, %cc0_1, %cc0_2] : tensor<32x512x256xf32>
    scf.yield %t_ins_cc0 : tensor<32x512x256xf32>
    }
    scf.yield %t_cc0_d2 : tensor<32x512x256xf32>
    }
    scf.yield %t_cc0_d1 : tensor<32x512x256xf32>
    }
    %t_cc1_d0 = scf.for %cc1_0 = %c0 to %in1_d0 step %c1 iter_args(%t_cc1_0 = %t_cc0_d0) -> (tensor<32x512x256xf32>) {
    %t_cc1_d1 = scf.for %cc1_1 = %c0 to %in1_d1 step %c1 iter_args(%t_cc1_1 = %t_cc1_0) -> (tensor<32x512x256xf32>) {
    %t_cc1_d2 = scf.for %cc1_2 = %c0 to %in1_d2 step %c1 iter_args(%t_cc1_2 = %t_cc1_1) -> (tensor<32x512x256xf32>) {
    %t_ax_off1 = arith.addi %in0_d2, %cc1_2 : index
    %cv1 = tensor.extract %in1[%cc1_0, %cc1_1, %cc1_2] : tensor<32x512x64xf32>
    %t_ins_cc1 = tensor.insert %cv1 into %t_cc1_2[%cc1_0, %cc1_1, %t_ax_off1] : tensor<32x512x256xf32>
    scf.yield %t_ins_cc1 : tensor<32x512x256xf32>
    }
    scf.yield %t_cc1_d2 : tensor<32x512x256xf32>
    }
    scf.yield %t_cc1_d1 : tensor<32x512x256xf32>
    }
    %cum_off_2 = arith.addi %in0_d2, %in1_d2 : index
    %t_cc2_d0 = scf.for %cc2_0 = %c0 to %in2_d0 step %c1 iter_args(%t_cc2_0 = %t_cc1_d0) -> (tensor<32x512x256xf32>) {
    %t_cc2_d1 = scf.for %cc2_1 = %c0 to %in2_d1 step %c1 iter_args(%t_cc2_1 = %t_cc2_0) -> (tensor<32x512x256xf32>) {
    %t_cc2_d2 = scf.for %cc2_2 = %c0 to %in2_d2 step %c1 iter_args(%t_cc2_2 = %t_cc2_1) -> (tensor<32x512x256xf32>) {
    %t_ax_off2 = arith.addi %cum_off_2, %cc2_2 : index
    %cv2 = tensor.extract %in2[%cc2_0, %cc2_1, %cc2_2] : tensor<32x512x64xf32>
    %t_ins_cc2 = tensor.insert %cv2 into %t_cc2_2[%cc2_0, %cc2_1, %t_ax_off2] : tensor<32x512x256xf32>
    scf.yield %t_ins_cc2 : tensor<32x512x256xf32>
    }
    scf.yield %t_cc2_d2 : tensor<32x512x256xf32>
    }
    scf.yield %t_cc2_d1 : tensor<32x512x256xf32>
    }
    %cum_off_3 = arith.addi %cum_off_2, %in2_d2 : index
    %t_cc3_d0 = scf.for %cc3_0 = %c0 to %in3_d0 step %c1 iter_args(%t_cc3_0 = %t_cc2_d0) -> (tensor<32x512x256xf32>) {
    %t_cc3_d1 = scf.for %cc3_1 = %c0 to %in3_d1 step %c1 iter_args(%t_cc3_1 = %t_cc3_0) -> (tensor<32x512x256xf32>) {
    %t_cc3_d2 = scf.for %cc3_2 = %c0 to %in3_d2 step %c1 iter_args(%t_cc3_2 = %t_cc3_1) -> (tensor<32x512x256xf32>) {
    %t_ax_off3 = arith.addi %cum_off_3, %cc3_2 : index
    %cv3 = tensor.extract %in3[%cc3_0, %cc3_1, %cc3_2] : tensor<32x512x64xf32>
    %t_ins_cc3 = tensor.insert %cv3 into %t_cc3_2[%cc3_0, %cc3_1, %t_ax_off3] : tensor<32x512x256xf32>
    scf.yield %t_ins_cc3 : tensor<32x512x256xf32>
    }
    scf.yield %t_cc3_d2 : tensor<32x512x256xf32>
    }
    scf.yield %t_cc3_d1 : tensor<32x512x256xf32>
    }
    return %t_cc3_d0 : tensor<32x512x256xf32>
  }
}

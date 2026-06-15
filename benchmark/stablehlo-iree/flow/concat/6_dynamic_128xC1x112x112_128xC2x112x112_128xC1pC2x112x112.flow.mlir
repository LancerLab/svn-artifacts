module @f_6_dynamic_128xC1x112x112_128xC2x112x112_128xC1pC2x112x112 {
  flow.executable private @f_6_dynamic_128xC1x112x112_128xC2x112x112_128xC1pC2x112x112_dispatch_0 {
    flow.executable.export public @f_6_dynamic_128xC1x112x112_128xC2x112x112_128xC1pC2x112x112_dispatch_0_generic_128xDx112x112_f32 workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1, %arg2
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_6_dynamic_128xC1x112x112_128xC2x112x112_128xC1pC2x112x112_dispatch_0_generic_128xDx112x112_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<128x?x112x112xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<128x?x112x112xf32>>, %arg2: index, %arg3: index, %arg4: index, %arg5: !flow.dispatch.tensor<writeonly:tensor<128x?x112x112xf32>>) {
        %0 = flow.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg3, 1 : index
        %2 = flow.dispatch.workload.ordinal %arg4, 2 : index
        %3 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<128x?x112x112xf32>>{%0}
        %4 = flow.dispatch.tie_shape %arg1 : !flow.dispatch.tensor<readonly:tensor<128x?x112x112xf32>>{%1}
        %5 = flow.dispatch.tie_shape %arg5 : !flow.dispatch.tensor<writeonly:tensor<128x?x112x112xf32>>{%2}
        %6 = flow.dispatch.tensor.load %3, offsets = [0, 0, 0, 0], sizes = [128, %0, 112, 112], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x?x112x112xf32>>{%0} -> tensor<128x?x112x112xf32>
        %7 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0, 0], sizes = [128, %1, 112, 112], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x?x112x112xf32>>{%1} -> tensor<128x?x112x112xf32>
        %8 = tensor.empty(%2) : tensor<128x?x112x112xf32>
        %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%8 : tensor<128x?x112x112xf32>) {
        ^bb0(%out: f32):
          %10 = linalg.index 0 : index
          %11 = linalg.index 2 : index
          %12 = linalg.index 3 : index
          %13 = linalg.index 1 : index
          %14 = arith.cmpi ult, %13, %0 : index
          %15 = scf.if %14 -> (f32) {
            %extracted = tensor.extract %6[%10, %13, %11, %12] : tensor<128x?x112x112xf32>
            scf.yield %extracted : f32
          } else {
            %16 = arith.subi %13, %0 : index
            %extracted = tensor.extract %7[%10, %16, %11, %12] : tensor<128x?x112x112xf32>
            scf.yield %extracted : f32
          }
          linalg.yield %15 : f32
        } -> tensor<128x?x112x112xf32>
        flow.dispatch.tensor.store %9, %5, offsets = [0, 0, 0, 0], sizes = [128, %2, 112, 112], strides = [1, 1, 1, 1] : tensor<128x?x112x112xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x?x112x112xf32>>{%2}
        return
      }
    }
  }
  func.func @f_6_dynamic_128xC1x112x112_128xC2x112x112_128xC1pC2x112x112(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
    %1 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<128x?x112x112xf32>{%0}
    %2 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
    %3 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<128x?x112x112xf32>{%2}
    %4 = arith.addi %0, %2 : index
    %5 = flow.dispatch @f_6_dynamic_128xC1x112x112_128xC2x112x112_128xC1pC2x112x112_dispatch_0::@f_6_dynamic_128xC1x112x112_128xC2x112x112_128xC1pC2x112x112_dispatch_0_generic_128xDx112x112_f32[%0, %2, %4](%1, %3, %0, %2, %4) : (tensor<128x?x112x112xf32>{%0}, tensor<128x?x112x112xf32>{%2}, index, index, index) -> tensor<128x?x112x112xf32>{%4}
    %6 = hal.tensor.export %5 "output 0" : tensor<128x?x112x112xf32>{%4} -> !hal.buffer_view
    return %6 : !hal.buffer_view
  }
}
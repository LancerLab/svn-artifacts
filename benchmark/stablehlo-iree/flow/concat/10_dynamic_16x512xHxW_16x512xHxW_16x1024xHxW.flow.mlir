module @f_10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW {
  flow.executable private @f_10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW_dispatch_0 {
    flow.executable.export public @f_10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW_dispatch_0_generic_16x1024xDxD_f32 workgroups(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1, %arg2, %arg3
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW_dispatch_0_generic_16x1024xDxD_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<16x512x?x?xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<16x512x?x?xf32>>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: !flow.dispatch.tensor<writeonly:tensor<16x1024x?x?xf32>>) {
        %c512 = arith.constant 512 : index
        %0 = flow.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg3, 1 : index
        %2 = flow.dispatch.workload.ordinal %arg4, 2 : index
        %3 = flow.dispatch.workload.ordinal %arg5, 3 : index
        %4 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<16x512x?x?xf32>>{%2, %3}
        %5 = flow.dispatch.tie_shape %arg1 : !flow.dispatch.tensor<readonly:tensor<16x512x?x?xf32>>{%0, %1}
        %6 = flow.dispatch.tie_shape %arg6 : !flow.dispatch.tensor<writeonly:tensor<16x1024x?x?xf32>>{%2, %3}
        %7 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0, 0], sizes = [16, 512, %2, %3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x512x?x?xf32>>{%2, %3} -> tensor<16x512x?x?xf32>
        %8 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0, 0], sizes = [16, 512, %0, %1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x512x?x?xf32>>{%0, %1} -> tensor<16x512x?x?xf32>
        %9 = tensor.empty(%2, %3) : tensor<16x1024x?x?xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%9 : tensor<16x1024x?x?xf32>) {
        ^bb0(%out: f32):
          %11 = linalg.index 0 : index
          %12 = linalg.index 2 : index
          %13 = linalg.index 3 : index
          %14 = linalg.index 1 : index
          %15 = arith.cmpi ult, %14, %c512 : index
          %16 = scf.if %15 -> (f32) {
            %extracted = tensor.extract %7[%11, %14, %12, %13] : tensor<16x512x?x?xf32>
            scf.yield %extracted : f32
          } else {
            %17 = arith.subi %14, %c512 : index
            %extracted = tensor.extract %8[%11, %17, %12, %13] : tensor<16x512x?x?xf32>
            scf.yield %extracted : f32
          }
          linalg.yield %16 : f32
        } -> tensor<16x1024x?x?xf32>
        flow.dispatch.tensor.store %10, %6, offsets = [0, 0, 0, 0], sizes = [16, 1024, %2, %3], strides = [1, 1, 1, 1] : tensor<16x1024x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x1024x?x?xf32>>{%2, %3}
        return
      }
    }
  }
  func.func @f_10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[2] : index
    %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[3] : index
    %2 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<16x512x?x?xf32>{%0, %1}
    %3 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[2] : index
    %4 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[3] : index
    %5 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<16x512x?x?xf32>{%3, %4}
    %6 = flow.dispatch @f_10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW_dispatch_0::@f_10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW_dispatch_0_generic_16x1024xDxD_f32[%3, %4, %0, %1](%2, %5, %3, %4, %0, %1) : (tensor<16x512x?x?xf32>{%0, %1}, tensor<16x512x?x?xf32>{%3, %4}, index, index, index, index) -> tensor<16x1024x?x?xf32>{%0, %1}
    %7 = hal.tensor.export %6 "output 0" : tensor<16x1024x?x?xf32>{%0, %1} -> !hal.buffer_view
    return %7 : !hal.buffer_view
  }
}
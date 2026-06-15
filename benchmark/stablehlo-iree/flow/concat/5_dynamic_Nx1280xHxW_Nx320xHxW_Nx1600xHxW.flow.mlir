module @f_5_dynamic_Nx1280xHxW_Nx320xHxW_Nx1600xHxW {
  flow.executable private @f_5_dynamic_Nx1280xHxW_Nx320xHxW_Nx1600xHxW_dispatch_0 {
    flow.executable.export public @f_5_dynamic_Nx1280xHxW_Nx320xHxW_Nx1600xHxW_dispatch_0_generic_Dx1600xDxD_f32 workgroups(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1, %arg2, %arg3, %arg4, %arg5
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_5_dynamic_Nx1280xHxW_Nx320xHxW_Nx1600xHxW_dispatch_0_generic_Dx1600xDxD_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<?x1280x?x?xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<?x320x?x?xf32>>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: !flow.dispatch.tensor<writeonly:tensor<?x1600x?x?xf32>>) {
        %c1280 = arith.constant 1280 : index
        %0 = flow.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg3, 1 : index
        %2 = flow.dispatch.workload.ordinal %arg4, 2 : index
        %3 = flow.dispatch.workload.ordinal %arg5, 3 : index
        %4 = flow.dispatch.workload.ordinal %arg6, 4 : index
        %5 = flow.dispatch.workload.ordinal %arg7, 5 : index
        %6 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x1280x?x?xf32>>{%3, %4, %5}
        %7 = flow.dispatch.tie_shape %arg1 : !flow.dispatch.tensor<readonly:tensor<?x320x?x?xf32>>{%0, %1, %2}
        %8 = flow.dispatch.tie_shape %arg8 : !flow.dispatch.tensor<writeonly:tensor<?x1600x?x?xf32>>{%3, %4, %5}
        %9 = flow.dispatch.tensor.load %6, offsets = [0, 0, 0, 0], sizes = [%3, 1280, %4, %5], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x1280x?x?xf32>>{%3, %4, %5} -> tensor<?x1280x?x?xf32>
        %10 = flow.dispatch.tensor.load %7, offsets = [0, 0, 0, 0], sizes = [%0, 320, %1, %2], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x320x?x?xf32>>{%0, %1, %2} -> tensor<?x320x?x?xf32>
        %11 = tensor.empty(%3, %4, %5) : tensor<?x1600x?x?xf32>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%11 : tensor<?x1600x?x?xf32>) {
        ^bb0(%out: f32):
          %13 = linalg.index 0 : index
          %14 = linalg.index 2 : index
          %15 = linalg.index 3 : index
          %16 = linalg.index 1 : index
          %17 = arith.cmpi ult, %16, %c1280 : index
          %18 = scf.if %17 -> (f32) {
            %extracted = tensor.extract %9[%13, %16, %14, %15] : tensor<?x1280x?x?xf32>
            scf.yield %extracted : f32
          } else {
            %19 = arith.subi %16, %c1280 : index
            %extracted = tensor.extract %10[%13, %19, %14, %15] : tensor<?x320x?x?xf32>
            scf.yield %extracted : f32
          }
          linalg.yield %18 : f32
        } -> tensor<?x1600x?x?xf32>
        flow.dispatch.tensor.store %12, %8, offsets = [0, 0, 0, 0], sizes = [%3, 1600, %4, %5], strides = [1, 1, 1, 1] : tensor<?x1600x?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x1600x?x?xf32>>{%3, %4, %5}
        return
      }
    }
  }
  func.func @f_5_dynamic_Nx1280xHxW_Nx320xHxW_Nx1600xHxW(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[2] : index
    %2 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[3] : index
    %3 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<?x1280x?x?xf32>{%0, %1, %2}
    %4 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
    %5 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[2] : index
    %6 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[3] : index
    %7 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<?x320x?x?xf32>{%4, %5, %6}
    %8 = flow.dispatch @f_5_dynamic_Nx1280xHxW_Nx320xHxW_Nx1600xHxW_dispatch_0::@f_5_dynamic_Nx1280xHxW_Nx320xHxW_Nx1600xHxW_dispatch_0_generic_Dx1600xDxD_f32[%4, %5, %6, %0, %1, %2](%3, %7, %4, %5, %6, %0, %1, %2) : (tensor<?x1280x?x?xf32>{%0, %1, %2}, tensor<?x320x?x?xf32>{%4, %5, %6}, index, index, index, index, index, index) -> tensor<?x1600x?x?xf32>{%0, %1, %2}
    %9 = hal.tensor.export %8 "output 0" : tensor<?x1600x?x?xf32>{%0, %1, %2} -> !hal.buffer_view
    return %9 : !hal.buffer_view
  }
}
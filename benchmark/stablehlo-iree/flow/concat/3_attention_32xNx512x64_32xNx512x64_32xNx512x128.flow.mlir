module @f_3_attention_32xNx512x64_32xNx512x64_32xNx512x128 {
  flow.executable private @f_3_attention_32xNx512x64_32xNx512x64_32xNx512x128_dispatch_0 {
    flow.executable.export public @f_3_attention_32xNx512x64_32xNx512x64_32xNx512x128_dispatch_0_generic_32xDx512x128_f32 workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_3_attention_32xNx512x64_32xNx512x64_32xNx512x128_dispatch_0_generic_32xDx512x128_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<32x?x512x64xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<32x?x512x64xf32>>, %arg2: index, %arg3: index, %arg4: !flow.dispatch.tensor<writeonly:tensor<32x?x512x128xf32>>) {
        %c64 = arith.constant 64 : index
        %0 = flow.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg3, 1 : index
        %2 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<32x?x512x64xf32>>{%1}
        %3 = flow.dispatch.tie_shape %arg1 : !flow.dispatch.tensor<readonly:tensor<32x?x512x64xf32>>{%0}
        %4 = flow.dispatch.tie_shape %arg4 : !flow.dispatch.tensor<writeonly:tensor<32x?x512x128xf32>>{%1}
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [32, %1, 512, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x?x512x64xf32>>{%1} -> tensor<32x?x512x64xf32>
        %6 = flow.dispatch.tensor.load %3, offsets = [0, 0, 0, 0], sizes = [32, %0, 512, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x?x512x64xf32>>{%0} -> tensor<32x?x512x64xf32>
        %7 = tensor.empty(%1) : tensor<32x?x512x128xf32>
        %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%7 : tensor<32x?x512x128xf32>) {
        ^bb0(%out: f32):
          %9 = linalg.index 0 : index
          %10 = linalg.index 1 : index
          %11 = linalg.index 2 : index
          %12 = linalg.index 3 : index
          %13 = arith.cmpi ult, %12, %c64 : index
          %14 = scf.if %13 -> (f32) {
            %extracted = tensor.extract %5[%9, %10, %11, %12] : tensor<32x?x512x64xf32>
            scf.yield %extracted : f32
          } else {
            %15 = arith.subi %12, %c64 : index
            %extracted = tensor.extract %6[%9, %10, %11, %15] : tensor<32x?x512x64xf32>
            scf.yield %extracted : f32
          }
          linalg.yield %14 : f32
        } -> tensor<32x?x512x128xf32>
        flow.dispatch.tensor.store %8, %4, offsets = [0, 0, 0, 0], sizes = [32, %1, 512, 128], strides = [1, 1, 1, 1] : tensor<32x?x512x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x?x512x128xf32>>{%1}
        return
      }
    }
  }
  func.func @f_3_attention_32xNx512x64_32xNx512x64_32xNx512x128(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
    %1 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<32x?x512x64xf32>{%0}
    %2 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
    %3 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<32x?x512x64xf32>{%2}
    %4 = flow.dispatch @f_3_attention_32xNx512x64_32xNx512x64_32xNx512x128_dispatch_0::@f_3_attention_32xNx512x64_32xNx512x64_32xNx512x128_dispatch_0_generic_32xDx512x128_f32[%2, %0](%1, %3, %2, %0) : (tensor<32x?x512x64xf32>{%0}, tensor<32x?x512x64xf32>{%2}, index, index) -> tensor<32x?x512x128xf32>{%0}
    %5 = hal.tensor.export %4 "output 0" : tensor<32x?x512x128xf32>{%0} -> !hal.buffer_view
    return %5 : !hal.buffer_view
  }
}
module @f_11_dynamic_64xN_40000x512_64xNx512 {
  flow.executable private @f_11_dynamic_64xN_40000x512_64xNx512_dispatch_0 {
    flow.executable.export public @f_11_dynamic_64xN_40000x512_64xNx512_dispatch_0_generic_64xDx512_f32 workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_11_dynamic_64xN_40000x512_64xNx512_dispatch_0_generic_64xDx512_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<64x?xi32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<40000x512xf32>>, %arg2: index, %arg3: !flow.dispatch.tensor<writeonly:tensor<64x?x512xf32>>) {
        %c0 = arith.constant 0 : index
        %c39999 = arith.constant 39999 : index
        %0 = flow.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<64x?xi32>>{%0}
        %2 = flow.dispatch.tie_shape %arg3 : !flow.dispatch.tensor<writeonly:tensor<64x?x512xf32>>{%0}
        %3 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [64, %0], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x?xi32>>{%0} -> tensor<64x?xi32>
        %4 = flow.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [40000, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<40000x512xf32>> -> tensor<40000x512xf32>
        %5 = tensor.empty(%0) : tensor<64x?x512xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} outs(%5 : tensor<64x?x512xf32>) {
        ^bb0(%out: f32):
          %7 = linalg.index 0 : index
          %8 = linalg.index 1 : index
          %9 = linalg.index 2 : index
          %extracted = tensor.extract %3[%7, %8] : tensor<64x?xi32>
          %10 = arith.index_cast %extracted : i32 to index
          %11 = arith.maxsi %10, %c0 : index
          %12 = arith.minsi %11, %c39999 : index
          %extracted_0 = tensor.extract %4[%12, %9] : tensor<40000x512xf32>
          linalg.yield %extracted_0 : f32
        } -> tensor<64x?x512xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [64, %0, 512], strides = [1, 1, 1] : tensor<64x?x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x?x512xf32>>{%0}
        return
      }
    }
  }
  func.func @f_11_dynamic_64xN_40000x512_64xNx512(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<40000x512xf32>
    %1 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
    %2 = hal.tensor.import %arg1 "input 1" : !hal.buffer_view -> tensor<64x?xi32>{%1}
    %3 = flow.dispatch @f_11_dynamic_64xN_40000x512_64xNx512_dispatch_0::@f_11_dynamic_64xN_40000x512_64xNx512_dispatch_0_generic_64xDx512_f32[%1](%2, %0, %1) : (tensor<64x?xi32>{%1}, tensor<40000x512xf32>, index) -> tensor<64x?x512xf32>{%1}
    %4 = hal.tensor.export %3 "output 0" : tensor<64x?x512xf32>{%1} -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
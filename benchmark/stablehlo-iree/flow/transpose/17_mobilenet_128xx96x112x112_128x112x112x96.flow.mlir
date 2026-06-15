module @f_17_mobilenet_128xx96x112x112_128x112x112x96 {
  flow.executable private @f_17_mobilenet_128xx96x112x112_128x112x112x96_dispatch_0 {
    flow.executable.export public @f_17_mobilenet_128xx96x112x112_128x112x112x96_dispatch_0_generic_128x12544x96_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @f_17_mobilenet_128xx96x112x112_128x112x112x96_dispatch_0_generic_128x12544x96_f32(%arg0: !flow.dispatch.tensor<readonly:tensor<128x96x12544xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<128x12544x96xf32>>) {
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [128, 96, 12544], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x96x12544xf32>> -> tensor<128x96x12544xf32>
        %1 = tensor.empty() : tensor<128x12544x96xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<128x96x12544xf32>) outs(%1 : tensor<128x12544x96xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<128x12544x96xf32>
        flow.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [128, 12544, 96], strides = [1, 1, 1] : tensor<128x12544x96xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x12544x96xf32>>
        return
      }
    }
  }
  func.func @f_17_mobilenet_128xx96x112x112_128x112x112x96(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<128x96x112x112xf32>
    %1 = flow.tensor.reshape %0 : tensor<128x96x112x112xf32> -> tensor<128x96x12544xf32>
    %2 = flow.dispatch @f_17_mobilenet_128xx96x112x112_128x112x112x96_dispatch_0::@f_17_mobilenet_128xx96x112x112_128x112x112x96_dispatch_0_generic_128x12544x96_f32(%1) : (tensor<128x96x12544xf32>) -> tensor<128x12544x96xf32>
    %3 = flow.tensor.reshape %2 : tensor<128x12544x96xf32> -> tensor<128x112x112x96xf32>
    %4 = hal.tensor.export %3 "output 0" : tensor<128x112x112x96xf32> -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}
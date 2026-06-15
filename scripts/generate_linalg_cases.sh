#!/bin/bash
# Generate linalg-level test cases for MLIR benchmark suite
# Each case has: correct + mismatched versions, both static and dynamic shapes
# These test the linalg verifier's compile-time detection (static) and
# the gap in dynamic-shape detection.
#
# Usage: ./scripts/generate_linalg_cases.sh

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LINALG_DIR="$WORKSPACE_ROOT/benchmark/mlir/cases_linalg"

mkdir -p "$LINALG_DIR"/{matmul,elemwise_add,conv2d,batch_matmul}

# ========================================================================
# matmul: linalg.matmul
# ========================================================================

# Static correct
cat > "$LINALG_DIR/matmul/01_static_128x1280_1280x64_correct.mlir" << 'EOF'
module {
  func.func @matmul_static_ok(%a: tensor<128x1280xf32>, %b: tensor<1280x64xf32>) -> tensor<128x64xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<128x64xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x64xf32>) -> tensor<128x64xf32>
    %r = linalg.matmul ins(%a, %b : tensor<128x1280xf32>, tensor<1280x64xf32>)
                        outs(%fill : tensor<128x64xf32>) -> tensor<128x64xf32>
    return %r : tensor<128x64xf32>
  }
}
EOF

# Static mismatch (K dim: 1280 vs 99)
cat > "$LINALG_DIR/matmul/02_static_128x1280_99x64_mismatch.mlir" << 'EOF'
module {
  func.func @matmul_static_bad(%a: tensor<128x1280xf32>, %b: tensor<99x64xf32>) -> tensor<128x64xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<128x64xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x64xf32>) -> tensor<128x64xf32>
    %r = linalg.matmul ins(%a, %b : tensor<128x1280xf32>, tensor<99x64xf32>)
                        outs(%fill : tensor<128x64xf32>) -> tensor<128x64xf32>
    return %r : tensor<128x64xf32>
  }
}
EOF

# Dynamic correct
cat > "$LINALG_DIR/matmul/03_dynamic_MxK_KxN_correct.mlir" << 'EOF'
module {
  func.func @matmul_dyn_ok(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.0 : f32
    %m = tensor.dim %a, %c0 : tensor<?x?xf32>
    %n = tensor.dim %b, %c1 : tensor<?x?xf32>
    %init = tensor.empty(%m, %n) : tensor<?x?xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
    %r = linalg.matmul ins(%a, %b : tensor<?x?xf32>, tensor<?x?xf32>)
                        outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %r : tensor<?x?xf32>
  }
}
EOF

# Mixed: static lhs K=1280, dynamic rhs K=?
cat > "$LINALG_DIR/matmul/04_mixed_128x1280_Qx64.mlir" << 'EOF'
module {
  func.func @matmul_mixed(%a: tensor<128x1280xf32>, %b: tensor<?x64xf32>) -> tensor<128x64xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<128x64xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x64xf32>) -> tensor<128x64xf32>
    %r = linalg.matmul ins(%a, %b : tensor<128x1280xf32>, tensor<?x64xf32>)
                        outs(%fill : tensor<128x64xf32>) -> tensor<128x64xf32>
    return %r : tensor<128x64xf32>
  }
}
EOF

# Additional static shapes from benchmark suite
for shape in "32x768_768x768" "64x256_256x128" "32x512_512x768" "16x1024_1024x4096"; do
  IFS='_' read -r lhs rhs <<< "$shape"
  IFS='x' read -r m k1 <<< "$lhs"
  IFS='x' read -r k2 n <<< "$rhs"
  cat > "$LINALG_DIR/matmul/static_${m}x${k1}_${k2}x${n}_correct.mlir" << EOFM
module {
  func.func @matmul(%a: tensor<${m}x${k1}xf32>, %b: tensor<${k2}x${n}xf32>) -> tensor<${m}x${n}xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<${m}x${n}xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<${m}x${n}xf32>) -> tensor<${m}x${n}xf32>
    %r = linalg.matmul ins(%a, %b : tensor<${m}x${k1}xf32>, tensor<${k2}x${n}xf32>)
                        outs(%fill : tensor<${m}x${n}xf32>) -> tensor<${m}x${n}xf32>
    return %r : tensor<${m}x${n}xf32>
  }
}
EOFM
  # Mismatch version (K-1)
  k_bad=$((k2 - 1))
  cat > "$LINALG_DIR/matmul/static_${m}x${k1}_${k_bad}x${n}_mismatch.mlir" << EOFM
module {
  func.func @matmul_bad(%a: tensor<${m}x${k1}xf32>, %b: tensor<${k_bad}x${n}xf32>) -> tensor<${m}x${n}xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<${m}x${n}xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<${m}x${n}xf32>) -> tensor<${m}x${n}xf32>
    %r = linalg.matmul ins(%a, %b : tensor<${m}x${k1}xf32>, tensor<${k_bad}x${n}xf32>)
                        outs(%fill : tensor<${m}x${n}xf32>) -> tensor<${m}x${n}xf32>
    return %r : tensor<${m}x${n}xf32>
  }
}
EOFM
done

# ========================================================================
# elemwise_add: linalg.add
# ========================================================================

cat > "$LINALG_DIR/elemwise_add/01_static_16x512_correct.mlir" << 'EOF'
module {
  func.func @add_ok(%a: tensor<16x512xf32>, %b: tensor<16x512xf32>) -> tensor<16x512xf32> {
    %init = tensor.empty() : tensor<16x512xf32>
    %r = linalg.add ins(%a, %b : tensor<16x512xf32>, tensor<16x512xf32>)
                    outs(%init : tensor<16x512xf32>) -> tensor<16x512xf32>
    return %r : tensor<16x512xf32>
  }
}
EOF

cat > "$LINALG_DIR/elemwise_add/02_static_16x512_16x99_mismatch.mlir" << 'EOF'
module {
  func.func @add_bad(%a: tensor<16x512xf32>, %b: tensor<16x99xf32>) -> tensor<16x512xf32> {
    %init = tensor.empty() : tensor<16x512xf32>
    %r = linalg.add ins(%a, %b : tensor<16x512xf32>, tensor<16x99xf32>)
                    outs(%init : tensor<16x512xf32>) -> tensor<16x512xf32>
    return %r : tensor<16x512xf32>
  }
}
EOF

cat > "$LINALG_DIR/elemwise_add/03_dynamic_correct.mlir" << 'EOF'
module {
  func.func @add_dyn(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %a, %c0 : tensor<?x?xf32>
    %n = tensor.dim %a, %c1 : tensor<?x?xf32>
    %init = tensor.empty(%m, %n) : tensor<?x?xf32>
    %r = linalg.add ins(%a, %b : tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %r : tensor<?x?xf32>
  }
}
EOF

# ========================================================================
# conv2d: linalg.conv_2d_nchw_fchw
# ========================================================================

cat > "$LINALG_DIR/conv2d/01_static_1x128x56x56_256x128x3x3_correct.mlir" << 'EOF'
module {
  func.func @conv_ok(%img: tensor<1x128x56x56xf32>, %ker: tensor<256x128x3x3xf32>) -> tensor<1x256x54x54xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<1x256x54x54xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x256x54x54xf32>) -> tensor<1x256x54x54xf32>
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%img, %ker : tensor<1x128x56x56xf32>, tensor<256x128x3x3xf32>)
         outs(%fill : tensor<1x256x54x54xf32>) -> tensor<1x256x54x54xf32>
    return %r : tensor<1x256x54x54xf32>
  }
}
EOF

cat > "$LINALG_DIR/conv2d/02_static_1x128x56x56_256x99x3x3_mismatch.mlir" << 'EOF'
module {
  func.func @conv_bad(%img: tensor<1x128x56x56xf32>, %ker: tensor<256x99x3x3xf32>) -> tensor<1x256x54x54xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<1x256x54x54xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x256x54x54xf32>) -> tensor<1x256x54x54xf32>
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%img, %ker : tensor<1x128x56x56xf32>, tensor<256x99x3x3xf32>)
         outs(%fill : tensor<1x256x54x54xf32>) -> tensor<1x256x54x54xf32>
    return %r : tensor<1x256x54x54xf32>
  }
}
EOF

cat > "$LINALG_DIR/conv2d/03_dynamic_correct.mlir" << 'EOF'
module {
  func.func @conv_dyn(%img: tensor<?x?x?x?xf32>, %ker: tensor<?x?x3x3xf32>) -> tensor<?x?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant 0.0 : f32
    %n = tensor.dim %img, %c0 : tensor<?x?x?x?xf32>
    %f = tensor.dim %ker, %c0 : tensor<?x?x?x?xf32>
    %h = tensor.dim %img, %c2 : tensor<?x?x?x?xf32>
    %w = tensor.dim %img, %c3 : tensor<?x?x?x?xf32>
    %c2i = arith.constant 2 : index
    %oh = arith.subi %h, %c2i : index
    %ow = arith.subi %w, %c2i : index
    %init = tensor.empty(%n, %f, %oh, %ow) : tensor<?x?x?x?xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    %r = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
         ins(%img, %ker : tensor<?x?x?x?xf32>, tensor<?x?x3x3xf32>)
         outs(%fill : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    return %r : tensor<?x?x?x?xf32>
  }
}
EOF

# ========================================================================
# batch_matmul: linalg.batch_matmul (covers batch_norm-like patterns)
# ========================================================================

cat > "$LINALG_DIR/batch_matmul/01_static_32x128x768_32x768x768_correct.mlir" << 'EOF'
module {
  func.func @bmm_ok(%a: tensor<32x128x768xf32>, %b: tensor<32x768x768xf32>) -> tensor<32x128x768xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<32x128x768xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<32x128x768xf32>) -> tensor<32x128x768xf32>
    %r = linalg.batch_matmul ins(%a, %b : tensor<32x128x768xf32>, tensor<32x768x768xf32>)
                              outs(%fill : tensor<32x128x768xf32>) -> tensor<32x128x768xf32>
    return %r : tensor<32x128x768xf32>
  }
}
EOF

cat > "$LINALG_DIR/batch_matmul/02_static_32x128x768_32x99x768_mismatch.mlir" << 'EOF'
module {
  func.func @bmm_bad(%a: tensor<32x128x768xf32>, %b: tensor<32x99x768xf32>) -> tensor<32x128x768xf32> {
    %cst = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<32x128x768xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<32x128x768xf32>) -> tensor<32x128x768xf32>
    %r = linalg.batch_matmul ins(%a, %b : tensor<32x128x768xf32>, tensor<32x99x768xf32>)
                              outs(%fill : tensor<32x128x768xf32>) -> tensor<32x128x768xf32>
    return %r : tensor<32x128x768xf32>
  }
}
EOF

cat > "$LINALG_DIR/batch_matmul/03_dynamic_BxMxK_BxKxN_correct.mlir" << 'EOF'
module {
  func.func @bmm_dyn(%a: tensor<?x?x?xf32>, %b: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.0 : f32
    %bat = tensor.dim %a, %c0 : tensor<?x?x?xf32>
    %m   = tensor.dim %a, %c1 : tensor<?x?x?xf32>
    %n   = tensor.dim %b, %c2 : tensor<?x?x?xf32>
    %init = tensor.empty(%bat, %m, %n) : tensor<?x?x?xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %r = linalg.batch_matmul ins(%a, %b : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
                              outs(%fill : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    return %r : tensor<?x?x?xf32>
  }
}
EOF

echo "Generated linalg-level test cases in: $LINALG_DIR"
find "$LINALG_DIR" -name "*.mlir" | sort
echo ""
echo "Total: $(find "$LINALG_DIR" -name "*.mlir" | wc -l) cases"

#!/usr/bin/env python3
"""Phase-0 settings derivation: extract operator contracts from benchmark/choreo/*/
into benchmark2/settings/<category>.md. Provenance recorded per case (source path
+ git blob hash). Source is never copied verbatim; only the operator contract
(name, tensor shapes, reference semantics) is emitted, per plan §2.2."""
import os, glob, hashlib

BASE = 'benchmark/choreo'
OUT = 'benchmark2/settings'
os.makedirs(OUT, exist_ok=True)

# operator -> (reference semantics, brief contract)
OPS = {
  'batch_norm': ("y[n,c,...] = (x[n,c,...] - mean_c)/sqrt(var_c+eps) * gamma[c] + beta[c]; "
                 "mean/var reduced over the non-channel dims (training mode, eps=1e-5)."),
  'concat':     ("y = concat(a, b, axis); shapes agree on all axes except the concat axis, "
                 "which sums."),
  'conv2d':     ("y = conv2d(x, w, stride, padding, dilation); standard NCHW 2-D convolution "
                 "(optionally strided/padded/dilated)."),
  'elemwise_add':("y = lhs + rhs; elementwise with broadcast (shapes equal or broadcastable)."),
  'embedding':  ("y = w[id]; gather rows of w (vocab_size x embed_dim) indexed by integer ids. "
                 "Output = id.shape + [embed_dim]."),
  'gelu':       ("y = 0.5*x*(1+erf(x/sqrt(2))); elementwise, output shape = input shape."),
  'layer_normalization': ("y = (x - mean)/sqrt(var+eps) * scale + bias; mean/var reduced over "
                          "the trailing (normalized) dims; scale/bias shaped to those dims."),
  'matmul':     ("y = lhs @ rhs; contract the inner dim K. Output = lhs.shape[:-1] + rhs.shape[-1:]."),
  'max_pool2d': ("y = windowed max over spatial dims (kernel/stride per case); NCHW."),
  'reduce_mean':("y = mean(x, axis); reduce one axis; output = x.shape with that axis dropped."),
  'relu':       ("y = max(x, 0); elementwise, output shape = input shape."),
  'reshape':    ("y = reshape(x, target); pure layout reinterpretation (element count preserved)."),
  'sigmoid':    ("y = 1/(1+exp(-x)); elementwise, output shape = input shape."),
  'softmax':    ("y = exp(x)/sum(exp(x), axis); normalized over the trailing axis; output = input shape."),
  'transpose':  ("y = permute(x, axes); output shape = permuted axes of input."),
}

def sig_of(path):
    for line in open(path):
        if '__co__' in line and '(' in line:
            return line.strip()
    return '(no __co__ signature found)'

def blob_hash(path):
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]

cats = sorted(d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d)))
for c in cats:
    files = sorted(glob.glob(os.path.join(BASE, c, '*.co')))
    sem = OPS.get(c, '(unspecified)')
    lines = []
    lines.append(f"# {c} — operator settings\n")
    lines.append(f"- **operator**: `{c}`")
    lines.append(f"- **reference semantics**: {sem}")
    lines.append(f"- **cases**: {len(files)} (derived from `benchmark/choreo/{c}/`; signatures only, not source)\n")
    lines.append("## Cases (provenance)\n")
    lines.append("| # | source file | blob | operator signature |")
    lines.append("|---|---|---|---|")
    for i, f in enumerate(files, 1):
        fn = os.path.basename(f)
        h = blob_hash(f)
        lines.append(f"| {i} | `benchmark/choreo/{c}/{fn}` | `{h}` | `{sig_of(f)}` |")
    lines.append("")
    with open(os.path.join(OUT, f'{c}.md'), 'w') as w:
        w.write("\n".join(lines))
    print(f"wrote settings/{c}.md ({len(files)} cases)")
print("DONE")

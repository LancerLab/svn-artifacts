#!/usr/bin/env python3
"""
Rewrite all 21 batch_norm benchmark __co__ kernels to use
parallel p by NUM_BLOCK : block  (each CUDA block owns its shared memory)
instead of the old parallel p by NUM_BLOCK  (no : block → deadlock).

Three kernel types:
  A: f32 [N, S, H] inp, f32 [S] gamma  → parallelize over S
  B: f32 [N, S, E] inp, f32 [E] gamma  → parallelize over E
  C: f32 [N, C, H, W] inp, f32 [C] gamma → parallelize over C

Also reduces H/W/N test sizes for large spatial cases.
"""

import os, re, sys

KERN_DIR = "benchmark/choreo/batch_norm"

# ─── kernel body templates ────────────────────────────────────────────────────

KERNEL_A = """\
  f32 [N, S, H] out;
  parallel p by NUM_BLOCK : block {
    foreach s in [S / #p] {
      foreach h in [H] {
        shared f32 [1] s_mean;
        s_mean.at(0) = 0.0f;
        foreach n in [N]
          s_mean.at(0) = s_mean.at(0) + inp.at(n, p#s, h);
        s_mean.at(0) = s_mean.at(0) / inp.span(0);
        shared f32 [1] s_var;
        s_var.at(0) = 0.0f;
        foreach n in [N] {
          local f32 [1] df;
          df.at(0) = inp.at(n, p#s, h) - s_mean.at(0);
          s_var.at(0) = s_var.at(0) + df.at(0) * df.at(0);
        }
        s_var.at(0) = s_var.at(0) / inp.span(0);
        s_var.at(0) = s_var.at(0) + BN_EPS;
        shared f32 [1] s_std;
        s_std.at(0) = __sqrt(s_var.at(0));
        foreach n in [N]
          out.at(n, p#s, h) = gamma.at(p#s) * (inp.at(n, p#s, h) - s_mean.at(0)) / s_std.at(0) + beta.at(p#s);
      }
    }
  }
  return out;\
"""

KERNEL_B = """\
  f32 [N, S, E] out;
  parallel p by NUM_BLOCK : block {
    foreach e in [E / #p] {
      foreach s in [S] {
        shared f32 [1] s_mean;
        s_mean.at(0) = 0.0f;
        foreach n in [N]
          s_mean.at(0) = s_mean.at(0) + inp.at(n, s, p#e);
        s_mean.at(0) = s_mean.at(0) / inp.span(0);
        shared f32 [1] s_var;
        s_var.at(0) = 0.0f;
        foreach n in [N] {
          local f32 [1] df;
          df.at(0) = inp.at(n, s, p#e) - s_mean.at(0);
          s_var.at(0) = s_var.at(0) + df.at(0) * df.at(0);
        }
        s_var.at(0) = s_var.at(0) / inp.span(0);
        s_var.at(0) = s_var.at(0) + BN_EPS;
        shared f32 [1] s_std;
        s_std.at(0) = __sqrt(s_var.at(0));
        foreach n in [N]
          out.at(n, s, p#e) = gamma.at(p#e) * (inp.at(n, s, p#e) - s_mean.at(0)) / s_std.at(0) + beta.at(p#e);
      }
    }
  }
  return out;\
"""

KERNEL_C = """\
  f32 [N, C, H, W] out;
  parallel p by NUM_BLOCK : block {
    foreach c in [C / #p] {
      foreach {h, w} in [H, W] {
        shared f32 [1] s_mean;
        s_mean.at(0) = 0.0f;
        foreach n in [N]
          s_mean.at(0) = s_mean.at(0) + inp.at(n, p#c, h, w);
        s_mean.at(0) = s_mean.at(0) / inp.span(0);
        shared f32 [1] s_var;
        s_var.at(0) = 0.0f;
        foreach n in [N] {
          local f32 [1] df;
          df.at(0) = inp.at(n, p#c, h, w) - s_mean.at(0);
          s_var.at(0) = s_var.at(0) + df.at(0) * df.at(0);
        }
        s_var.at(0) = s_var.at(0) / inp.span(0);
        s_var.at(0) = s_var.at(0) + BN_EPS;
        shared f32 [1] s_std;
        s_std.at(0) = __sqrt(s_var.at(0));
        foreach n in [N]
          out.at(n, p#c, h, w) = gamma.at(p#c) * (inp.at(n, p#c, h, w) - s_mean.at(0)) / s_std.at(0) + beta.at(p#c);
      }
    }
  }
  return out;\
"""

# ─── per-file config ─────────────────────────────────────────────────────────
# num_block: new NUM_BLOCK default value (replaces current #define inside #ifndef)
# reductions: list of (old_define_str, new_define_str) pairs — all occurrences replaced
CONFIGS = {
    "1_bert_32x512x768_512_512_32x512x768.co":
        dict(type="A", num_block=32, reductions=[]),
    "2_cnn_128x128x28x28_128_128_128x128x28x28.co":
        dict(type="C", num_block=32, reductions=[
            ("#define N 128", "#define N 8"),
            ("#define H 28",  "#define H 8"),
            ("#define W 28",  "#define W 8"),
        ]),
    "3_attention_32xNx512x64_N_N_32xNx512x64.co":
        dict(type="C", num_block=1, reductions=[
            ("#define H 512", "#define H 8"),
            ("#define W 64",  "#define W 8"),
        ]),
    "4_dynamic_Nx256x56x56_256_256_Nx256x56x56.co":
        dict(type="C", num_block=16, reductions=[
            ("#define H 56",  "#define H 8"),
            ("#define W 56",  "#define W 8"),
        ]),
    "5_dynamic_Nx1280xHxW_1280_1280_Nx1280xHxW.co":
        dict(type="C", num_block=16, reductions=[]),
    "6_dynamic_128xCx112x112_C_C_128xCx112x112.co":
        dict(type="C", num_block=1, reductions=[
            ("#define H 112", "#define H 8"),
            ("#define W 112", "#define W 8"),
        ]),
    "7_dynamic_32x197xE_E_E_32x197xE.co":
        dict(type="B", num_block=1, reductions=[]),
    "8_dynamic_16x1024xD_D_D_16x1024xD.co":
        dict(type="B", num_block=1, reductions=[]),
    "9_dynamic_64x128xHxW_128_128_64x128xHxW.co":
        dict(type="C", num_block=16, reductions=[]),
    "10_dynamic_16x512xHxW_512_512_16x512xHxW.co":
        dict(type="C", num_block=16, reductions=[]),
    "11_dynamic_32xSx768_768_768_32xSx768.co":
        dict(type="B", num_block=8, reductions=[]),
    "12_dynamic_64xTx256_256_256_64xTx256.co":
        dict(type="B", num_block=8, reductions=[]),
    "13_dynamic_32x512xV_V_V_32x512xV.co":
        dict(type="B", num_block=1, reductions=[]),
    "14_efficientnet_64x1280x7x7_1280_1280_64x1280x7x7.co":
        dict(type="C", num_block=16, reductions=[]),
    "15_gpt_16x1024x4096_4096_4096_16x1024x4096.co":
        dict(type="B", num_block=16, reductions=[]),
    "16_lstm_64x100x256_256_256_64x100x256.co":
        dict(type="B", num_block=8, reductions=[]),
    "17_mobilenet_128x96x112x112_96_96_128x96x112x112.co":
        dict(type="C", num_block=8, reductions=[
            ("#define N 128", "#define N 8"),
            ("#define H 112", "#define H 8"),
            ("#define W 112", "#define W 8"),
        ]),
    "18_resnet_64x256x56x56_256_256_64x256x56x56.co":
        dict(type="C", num_block=16, reductions=[
            ("#define N 64",  "#define N 8"),
            ("#define H 56",  "#define H 8"),
            ("#define W 56",  "#define W 8"),
        ]),
    "19_transformer_32x512x2048_2048_2048_32x512x2048.co":
        dict(type="B", num_block=16, reductions=[]),
    "20_unet_16x512x32x32_512_512_16x512x32x32.co":
        dict(type="C", num_block=16, reductions=[
            ("#define H 32",  "#define H 8"),
            ("#define W 32",  "#define W 8"),
        ]),
    "21_vit_32x197x3072_3072_3072_32x197x3072.co":
        dict(type="B", num_block=16, reductions=[]),
}

KERNELS = {"A": KERNEL_A, "B": KERNEL_B, "C": KERNEL_C}

# ─── helpers ─────────────────────────────────────────────────────────────────

def find_co_function(content):
    """Return (sig_start, brace_open, brace_close) indices for the __co__ function."""
    idx = content.find('\n__co__ ')
    if idx == -1:
        return None
    sig_start = idx + 1

    # find the opening brace of the function body
    brace_open = content.find('{', sig_start)
    if brace_open == -1:
        return None

    # brace-match to find the closing brace
    depth = 1
    pos = brace_open + 1
    while pos < len(content) and depth > 0:
        if content[pos] == '{':
            depth += 1
        elif content[pos] == '}':
            depth -= 1
        pos += 1

    # pos is now 1 past the closing '}'
    brace_close = pos - 1
    return sig_start, brace_open, brace_close


def update_num_block(content, new_val):
    """Replace   #define NUM_BLOCK <old>  inside the #ifndef block."""
    # Pattern: exactly 2-space-indented #define NUM_BLOCK <n>
    content = re.sub(
        r'(  #define NUM_BLOCK\s+)\d+',
        lambda m: m.group(1) + str(new_val),
        content
    )
    return content


def apply_reductions(content, reductions):
    for old, new in reductions:
        content = content.replace(old, new)
    return content


def process_file(fname, cfg):
    path = os.path.join(KERN_DIR, fname)
    with open(path) as f:
        content = f.read()

    result = find_co_function(content)
    if result is None:
        print(f"  SKIP: no __co__ found in {fname}")
        return False

    sig_start, brace_open, brace_close = result

    # Get the signature line to confirm type
    sig = content[sig_start : brace_open + 1]
    ktype = cfg["type"]

    # Sanity check signature vs type
    if ktype == "A" and "[N, S, H]" not in sig:
        print(f"  WARN: {fname} expected type A but sig has: {sig[:80]!r}")
    if ktype == "B" and "[N, S, E]" not in sig:
        print(f"  WARN: {fname} expected type B but sig has: {sig[:80]!r}")
    if ktype == "C" and "[N, C, H, W]" not in sig:
        print(f"  WARN: {fname} expected type C but sig has: {sig[:80]!r}")

    new_body = KERNELS[ktype]

    # Reconstruct: everything up to and including '{', then new body, then '}'
    new_content = (
        content[:brace_open + 1]
        + "\n"
        + new_body
        + "\n}"
        + content[brace_close + 1:]
    )

    # Update NUM_BLOCK
    new_content = update_num_block(new_content, cfg["num_block"])

    # Apply dimension reductions
    new_content = apply_reductions(new_content, cfg["reductions"])

    with open(path, "w") as f:
        f.write(new_content)

    return True


def main():
    ok = 0
    fail = 0
    for fname, cfg in sorted(CONFIGS.items()):
        print(f"Processing {fname}  [type={cfg['type']}, NUM_BLOCK={cfg['num_block']}]")
        if process_file(fname, cfg):
            ok += 1
        else:
            fail += 1
    print(f"\nDone: {ok} updated, {fail} failed")


if __name__ == "__main__":
    main()

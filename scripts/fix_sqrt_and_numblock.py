#!/usr/bin/env python3
"""
Fix all issues in the rewritten batch_norm and layer_norm kernels:
  1. Replace `__sqrt(expr + BN_EPS)` with local-var approach that avoids 1e-5f
     (Choreo parser rejects `1e-5f` as a literal in DSL code)
  2. Add `#define NUM_BLOCK N` to embedding files that are missing it
  3. Fix layer_norm files: replace `__sqrt(s_var.at(0) + 1e-5f)` with local-var
"""

import os, re
import sys

# ─── Fix 1 & 3: replace sqrt(expr + eps_literal) across batch_norm and layer_norm

def fix_sqrt_eps(content):
    """
    Fix three issues in rewritten DSL kernels:
    1. `local f32 df; df = expr;`  →  `local f32 [1] df; df.at(0) = expr;`
       followed by `df * df`  →  `df.at(0) * df.at(0)`
    2. `s_std.at(0) = __sqrt(s_var.at(0) + <eps>);`  →
         add eps to s_var before calling sqrt:
           s_var.at(0) = s_var.at(0) + 0.00001;
           s_std.at(0) = __sqrt(s_var.at(0));
    3. Remove the `local f32 _var_eps; _var_eps = ...; s_std.at(0) = __sqrt(_var_eps);`
       pattern injected by the previous fix run.
    """
    # Fix 1a: `local f32 df;`  →  `local f32 [1] df;`
    content = re.sub(
        r'\blocal f32 df\s*;',
        'local f32 [1] df;',
        content
    )
    # Fix 1b: `df = expr - s_mean.at(0);`  →  `df.at(0) = expr - s_mean.at(0);`
    content = re.sub(
        r'\bdf = (.*?- s_mean\.at\(0\));',
        r'df.at(0) = \1',
        content
    )
    # Fix 1c: `df * df`  →  `df.at(0) * df.at(0)`
    content = re.sub(r'\bdf \* df\b', 'df.at(0) * df.at(0)', content)

    # Fix 2: `s_std.at(0) = __sqrt(s_var.at(0) + <eps>);` where eps is a literal or BN_EPS
    def replace_sqrt_with_eps(m):
        indent = m.group(1)
        eps_token = m.group(2).strip()
        # Replace with two statements: add eps to s_var, then sqrt
        return (
            f"{indent}s_var.at(0) = s_var.at(0) + 0.00001;\n"
            f"{indent}s_std.at(0) = __sqrt(s_var.at(0));"
        )

    content = re.sub(
        r'^( +)s_std\.at\(0\) = __sqrt\(s_var\.at\(0\) \+ (BN_EPS|1e-5f?|0\.00001f?)\);',
        replace_sqrt_with_eps,
        content,
        flags=re.MULTILINE
    )

    # Fix 3: Remove old injected _var_eps lines (from previous fix_sqrt_and_numblock run)
    # Pattern:
    #   local f32 _var_eps;\n
    #   _var_eps = s_var.at(0) + <eps>;\n
    #   s_std.at(0) = __sqrt(_var_eps);
    content = re.sub(
        r'\s*local f32 _var_eps;\s*\n\s*_var_eps = s_var\.at\(0\) \+ [^;]+;\s*\n(\s*)s_std\.at\(0\) = __sqrt\(_var_eps\);',
        r'\n\1s_std.at(0) = __sqrt(s_var.at(0));',
        content
    )

    return content


# ─── Fix 2: add NUM_BLOCK define to embedding files
EMBEDDING_DIR = "benchmark/choreo/embedding"

# Batch sizes per file (from filename pattern or first-line dimension)
# For embedding, NUM_BLOCK = bs (first dim) is usually large; use a sensible smaller value
# We'll pick NUM_BLOCK = 8 as default (safe and parallel)
EMBED_NUM_BLOCK_DEFAULT = 8


def add_num_block_to_embedding(content, num_block):
    """Prepend #ifndef NUM_BLOCK / #define NUM_BLOCK / #endif after #include lines."""
    if '#define NUM_BLOCK' in content or '#ifndef NUM_BLOCK' in content:
        return content  # already has it

    # Insert after the last #include line
    last_include_end = -1
    for m in re.finditer(r'^#include\s.*$', content, re.MULTILINE):
        last_include_end = m.end()

    if last_include_end == -1:
        # Prepend
        insert_at = 0
    else:
        insert_at = last_include_end + 1  # after the newline

    block = (
        "\n#ifndef NUM_BLOCK\n"
        f"  #define NUM_BLOCK {num_block}\n"
        "#endif\n"
        "#ifndef NUM_THREAD\n"
        "  #define NUM_THREAD 1\n"
        "#endif\n"
    )
    return content[:insert_at] + block + content[insert_at:]


# ─── Main

def process_dir(dirpath, fix_sqrt=True, fix_embed=False):
    changed = 0
    for fname in sorted(os.listdir(dirpath)):
        if not fname.endswith('.co') or fname.endswith('.bak'):
            continue
        path = os.path.join(dirpath, fname)
        with open(path) as f:
            content = f.read()

        new_content = content
        if fix_sqrt:
            new_content = fix_sqrt_eps(new_content)
        if fix_embed:
            new_content = add_num_block_to_embedding(new_content, EMBED_NUM_BLOCK_DEFAULT)

        if new_content != content:
            with open(path, 'w') as f:
                f.write(new_content)
            print(f"  Updated: {fname}")
            changed += 1
        else:
            print(f"  No change: {fname}")

    return changed


if __name__ == '__main__':
    print("=== Fixing batch_norm kernels (sqrt+eps) ===")
    n = process_dir("benchmark/choreo/batch_norm", fix_sqrt=True, fix_embed=False)
    print(f"  {n} files changed")

    print("\n=== Fixing layer_norm kernels (sqrt+eps) ===")
    n = process_dir("benchmark/choreo/layer_normalization", fix_sqrt=True, fix_embed=False)
    print(f"  {n} files changed")

    print("\n=== Adding NUM_BLOCK to embedding files ===")
    n = process_dir("benchmark/choreo/embedding", fix_sqrt=False, fix_embed=True)
    print(f"  {n} files changed")

    print("\nAll done.")

#!/usr/bin/env python3
"""Why did 37 admissible E1 mutants come back `outcome=never`?

E1's headline number is uncomfortable: of the 72 `manifest=corrupts` mutants
(the admissible population — the 48 `noop` ones are discarded per specs §11.1),
choreo detected 35 before device execution and 37 came back `never`. That breaks
S2's `never = 0` assertion, which is the abstract's "210/210 caught before device
execution" claim restated on this mutant set.

This script exists because **`never` is not one thing**. Reporting it as a single
count — "choreo missed 37 bugs" — would be false in a way that matters: most of
the 37 are cases where choreo's assessor *did* identify the violation and then
lost it downstream. The causes are mechanistically distinct, they have different
owners (paper framing vs croqtile codegen), and they imply different fixes. So
this script attributes every one of the 37 to exactly one cause, with the ledger
evidence attached, and writes the result to `raw/never_attribution.json`.

--------------------------------------------------------------------------
THE FIVE CAUSES, AND THE EVIDENCE THAT SEPARATES THEM
--------------------------------------------------------------------------
Each cause is decided by a specific, checkable observation. They are tested in
this order, and the first match wins, so a mutant is never double-counted:

  C1  COST-FILTER SUPPRESSION
      The ledger carries an obligation for the mutated access with
      `outcome: runtime` and `enabled: false`. choreo's assessor reached the
      right verdict — "this needs a runtime guard" — but the guard's cost tier
      (medium/high) exceeds the default `-rtc` level (which measures as
      `entry`/`low`, byte-identical), so the cost filter dropped it.
      `--stats` shows this directly as "Runtime assertions disabled by cost
      filter". This is an EMISSION POLICY, not a soundness failure.

  C2  HOISTING-PLACEMENT DEFECT
      The guard IS emitted (at `-rtc=all`) but the assertion-hoisting pass moves
      it out of the loop body to a point AFTER the induction variable has been
      reset (`__iv_k = 0;`), so it evaluates `0 + 1 < 768` — vacuously true, can
      never fire. Detected by re-running with `--disable-assert-hoist`: if
      choreo's own guard then fires, the defect is the hoist, not the assessor.
      **This is a croqtile codegen bug.** Requires `--execute` (nvcc), so it is
      only tested when that flag is passed.

  C3  LOWER-BOUND OMISSION
      The ledger assessed the mutated index but generated ONLY the upper bound
      (`should be less than N`) and not the lower bound
      (`should be greater than or equal to 0`) — and the lower bound is the one
      that fails. Observed on `q - 1` at `q == 0`: every *literal* `0` index in
      the same kernel got both bounds, the mutated expression got one.
      **Also a croqtile soundness bug**, in obligation generation rather than
      placement.

  C4  NOT ASSESSED
      The mutated array does not appear in the ledger at all. choreo generated no
      obligation for that access, so there was nothing to suppress and nothing to
      hoist. Observed on `dma.copy` destination chunks (`out.chunkat(...)`) and
      on `with index` tile counts — surfaces the assessor does not model.
      A COVERAGE GAP, and the honest reading of `never` for these.

  C5  OUT OF SCOPE
      The mutation changes a VALUE, not a bound: `k -> 0` (in bounds, wrong
      element), a duplicated write, a shortened loop. There is no index-bound
      violation to detect, so `never` is the CORRECT verdict. specs §4 puts
      numeric correctness out of scope; these mutants should not have been
      counted as injections at all, and this script says so rather than letting
      them inflate the miss count.

--------------------------------------------------------------------------
WHAT THIS DOES NOT DO
--------------------------------------------------------------------------
It does not re-run E1 or rewrite `raw/e1_mutant_records.json`. The E1 records are
the measurement; this is an attribution pass over them. It also does not change
any statistic — `stats.py` still reports S2 exactly as measured (35/72, never=37,
assertion FAIL). This script explains the 37; it does not make them go away.

The `--execute` arm invokes nvcc (~10 s per mutant) and runs kernels on the GPU.
It is opt-in because C2 is the only cause that needs it, and because a correctness
lane may share the GPU freely (manifest §6) but should still not be launched
blindly. Without `--execute`, C2 candidates are reported as `untested` rather
than silently folded into another cause.

FEEDS  the coordinator report on S2, and (if C2/C3 are confirmed as systematic)
       a croqtile bug report. Does not feed any `stats.json` field directly.
"""

import argparse
import collections
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.dirname(HERE)
REPO = os.path.dirname(B2)
RAW = os.path.join(HERE, "raw")

# Where probe_execute() dumps the RAW combined stdout+stderr of each arm, keyed
# by mutant and arm. Set by main() from --raw-logs; None means don't persist.
#
# WHY THIS EXISTS. The verdict labels in the attribution artifact are a
# projection of the raw text through the three-source regexes above. When those
# regexes were wrong (matching only 1 of 7 assessment templates, and taking the
# oracle by negation) the stored `detail` field was not enough to re-derive the
# correct label: for an `oracle-only` row RE_ORACLE's lookahead matched a
# ZERO-WIDTH prefix, so `detail` recorded "choreo assertion failed: " and the
# actual message was gone. The only way to reclassify was to re-run nvcc for all
# 37 mutants -- ~12 s each, and it competes for the GPU with whatever else is on
# the host. Persisting the raw text costs a few KB per mutant and makes every
# future classifier correction a pure re-projection over stored evidence.
PROBE_RAW_DIR = None
MUTANTS = os.path.join(HERE, "mutants")
CHOREO = os.path.join(REPO, "croqtile", "build-release", "choreo")

CAP = "--max-local-mem-capacity=2000000"
LEDGER_FLAGS = ["-gs", "--stats", "-es", CAP, "-t", "cute"]
EXEC_FLAGS = ["-gs", "-t", "cute", "-kt", CAP, "-rtc=all"]

# ---------------------------------------------------------------------------
# THREE sources of assertion output, split across TWO prefixes and TWO streams.
# ---------------------------------------------------------------------------
# run_e1.py:classify_log carries the full enumeration with file:line citations.
# The short version, and the part that was gotten wrong here first:
#
#   A  generated assessments (lib/semacheck.cpp CreateAssessment, emitted by
#      CuteCodeGen::Emit{Pre,Post}SiteAssertions at cute_codegen.cpp:11789/11801,
#      both of which begin `if (CCtx().DisableRuntimeCheck()) return;`). GATED,
#      so this is the assessor and counts as a detection. A has TWO emission
#      forms depending on whether the site is host or device code:
#        A-host    runtime_check  -> STDERR, "choreo runtime check failed: "
#        A-device  choreo_assert  -> STDOUT, "choreo assertion failed: "
#      (device uses printf because, per cute_codegen.cpp:11805, "std::cerr is
#      not available on device"). BOTH append the source location into the
#      message (`ar.message << ", " << ar.loc`), so both end with
#      `, <path>:<line>.<col>` -- that suffix, not a template list, is the
#      reliable fingerprint.
#   B  runtime-library bounds check (ArrayProxy::operator[],
#      runtime/choreo.h:394/405/782/791/884) -- bare "Index out of bounds" on
#      stdout, no location suffix, NOT gated (measured: det.sh=12 off.sh=12 on
#      M2.s4.ln1.caller.out). Not assessor output.
#   C  §7 oracle (user-written `assert` BIF, cute_codegen.cpp:8359) -- stdout,
#      no location suffix, NOT gated. The harness's own numeric reference check.
#
# THE BUG THIS REPLACES, in two successive versions.
# Version 1 was
#     RE_CHOREO_GUARD = "...: The \w+ index .* of element access"
#     RE_ORACLE       = "...: (?!The \w+ index)"
# i.e. A matched by ONE template and everything else oracle by NEGATION. Two
# consequences, both flattering the harness and penalising choreo: an A message
# from any other template fell through to oracle-only, so a real detection was
# recorded as a miss; and a B firing was absorbed into oracle-only, hiding that
# the manifest verdict rested on choreo-provided code rather than independent
# ground truth. B is reachable from inside the oracle arm because the §7
# comparison indexes through ArrayProxy (`out[i][j][k]`,
# layer_normalization/1_bert:90). Measured: 2 detector-arm and 4 oracle-arm rows
# in the retained E1 logs are B.
# Version 2 fixed the negation but still matched A against the
# "choreo assertion failed: " prefix with seven enumerated templates. That
# regex matched ZERO lines in 160 retained run logs, because in this suite A
# only ever emits through the host form, i.e. the "choreo runtime check
# failed: " prefix. A template list is also inherently incomplete -- it covers
# only the assessments this semacheck version happens to create -- so match the
# location suffix the emitter itself appends instead.
#
# Source A on stdout (device form). Currently unexercised in this suite: zero of
# the 79 "choreo assertion failed" lines in the retained logs carry a location
# suffix. Kept because it is what the codegen does on device sites.
RE_CHOREO_GUARD = re.compile(
    r"choreo assertion failed: .*\.co:\d+\.\d+\s*$", re.MULTILINE
)
# Source A on stderr (host form), family 1: carries the location suffix.
# Observed 6 times in the retained logs, e.g.
#   "choreo runtime check failed: The 3rd index ` (q + 1) ` of element access
#    'inp_s' should be less than 64, .../M1.s2.rl11.read__....co:85.48"
RE_ENTRY_CHECK = re.compile(
    r"choreo runtime check failed: .*\.co:\d+\.\d+\s*$", re.MULTILINE
)
# Source A on stderr, family 2: NO location suffix, so RE_ENTRY_CHECK misses it.
# `shape inconsistent on the <ord> parameter ('<name>', dim: <n>): expect: <n>,
# but got <n>.` is emitted via runtime_check at cute_codegen.cpp:9965+ and is
# nonetheless GATED -- measured over 20 mutants, a base build has 5 such sites
# and a --disable-runtime-check build has 0. It is the commonest detection in
# this suite: 10 of the 16 source-A log lines.
RE_SHAPE_CHECK = re.compile(
    r"choreo runtime check failed: shape inconsistent on the \w+ parameter \("
)
# UNGATED LIBRARY SANITY CHECKS. These share source A's host prefix but are NOT
# assessments: they live in runtime/choreo.h, survive --disable-runtime-check,
# and report harness or allocator faults rather than a static-analysis verdict.
# The complete closed set (enumerated from runtime/ and cross-checked by diffing
# base vs --disable-runtime-check codegen over 20 mutants) is seven literals,
# all beginning `at::Tensor ` or `[choreo-rt]`. Treating the bare prefix as a
# detection would credit an allocation failure or a non-contiguous input tensor
# to the assessor. No such line occurs in the 160 retained logs, so this is a
# latent bug rather than a measured error -- but it is the same defect class as
# the negation above and gets the same treatment: enumerate positively.
RE_LIB_SANITY = re.compile(
    r"choreo runtime check failed: (?:at::Tensor |\[choreo-rt\])"
)
# Source B. Anchored to end-of-line so it cannot swallow an assessment's longer
# "Index <n> is out of bounds of the <ord> dimension of array ...,
# <path>:<line>.<col>" -- the two share a prefix and diverge after "bounds".
RE_RUNTIME_LIB = re.compile(r"choreo assertion failed: Index out of bounds\s*$",
                            re.MULTILINE)
# Source C, matched POSITIVELY against the closed vocabulary actually present in
# benchmark/choreo (85 "values are not equal.", 21 "concat mismatch",
# 19 "batch_norm mismatch", 16 "reduce_mean mismatch", 4 "error",
# 2 "Test Failed", 1 "actual != expect", 1 "sampled verification failed").
# Positive matching matters: a negation turns any message nobody anticipated
# into an oracle verdict, which is precisely the silent mis-attribution above.
RE_ORACLE = re.compile(
    r"choreo assertion failed: (?:"
    r"values are not equal\.|concat mismatch|batch_norm mismatch"
    r"|reduce_mean mismatch|Test Failed|actual != expect"
    r"|sampled verification failed|error"
    r")"
)
# An assertion whose message matches none of the vocabularies above. Kept
# separate from `oracle` so an unrecognised choreo message cannot be credited to
# the harness by default.
RE_ANY_ASSERT = re.compile(r"choreo assertion failed")
# A `runtime check failed` line that is neither a known assessment nor a known
# library check. Also kept separate, for the same reason.
RE_ANY_RTC = re.compile(r"choreo runtime check failed")

# probe_execute() verdicts that mean "source A fired", i.e. a DETECTION. Named
# here rather than repeated as an inline tuple at each consumer, because the
# inline form is how a newly added detection verdict gets silently dropped from
# `hoisting_defect_confirmed` and from the GUARD mapping: the tuple was written
# when only two source-A forms were known, and `shape-check-fired` would have
# been left out of both had it been added inline again.
DETECTION_VERDICTS = ("choreo-guard-fired", "entry-check-fired",
                      "shape-check-fired")

# A ledger obligation message looks like:
#   "The 3rd index ` (k + 1) ` of element access 'out' should be less than 768"
RE_OBL = re.compile(
    r"The (?P<ord>\w+) index `(?P<idx>.*?)` of element access '(?P<arr>[^']*)' "
    r"should be (?P<bound>.*)$")

UPPER = "less than"
LOWER = "greater than or equal to 0"


# --------------------------------------------------------------------------
# mutant discovery
# --------------------------------------------------------------------------

def mutant_path(mid):
    """`mutants/<CLASS>/<category>/<mutant_id>__<case>.co`.

    The `__<case>` suffix means a bare `-name "<mid>.co"` lookup finds nothing;
    glob on `"<mid>__*.co"` instead. Two directory levels, not one.
    """
    hits = glob.glob(os.path.join(MUTANTS, "*", "*", f"{mid}__*.co"))
    return hits[0] if hits else None


def workdir_for(src):
    """Copy the mutant to `k.co` in a fresh temp dir, plus its local headers.

    Two reasons for the copy. (1) The generated `.cu` filename derives from the
    INPUT filename, so a mutant named `M1.s2.ln1.out__1_bert_...co` produces
    `__choreo_cute_M1.s2.ln1.out__1_bert_....cu`; renaming to `k.co` makes the
    capture path predictable. (2) The generated script passes `-I<dir of the .co
    file>`, so `common.h`/`common.hpp` must sit next to it or the compile fails
    with "fatal error: common.hpp: No such file or directory".
    """
    t = tempfile.mkdtemp(prefix="never_")
    shutil.copy(src, os.path.join(t, "k.co"))
    for h in glob.glob(os.path.join(os.path.dirname(src), "*")):
        if not h.endswith(".co"):
            shutil.copy(h, t)
    return t


def run_ledger(src):
    """Return (obligations, stats_text) or (None, err)."""
    t = workdir_for(src)
    try:
        lp = os.path.join(t, "l.json")
        p = subprocess.run(
            [CHOREO] + LEDGER_FLAGS + [f"--dump-ledger={lp}",
                                       os.path.join(t, "k.co"),
                                       "-o", os.path.join(t, "k.sh")],
            capture_output=True, text=True, cwd=REPO)
        if not os.path.exists(lp):
            return None, f"no ledger emitted (rc={p.returncode})"
        with open(lp) as f:
            return json.load(f).get("obligations", []), p.stdout + p.stderr
    finally:
        shutil.rmtree(t, ignore_errors=True)


def run_execute(src, extra_flags):
    """Compile + run a mutant, returning (rc, combined_output).

    `stdbuf -o0 -e0` is mandatory: the abort path discards buffered stdout, and
    the attribution message is printed on stdout immediately before the abort.
    Without it a detection looks like a silent crash.
    """
    t = workdir_for(src)
    try:
        sh = os.path.join(t, "k.sh")
        p = subprocess.run(
            [CHOREO] + EXEC_FLAGS + extra_flags + [os.path.join(t, "k.co"), "-o", sh],
            capture_output=True, text=True, cwd=REPO)
        if p.returncode != 0 or not os.path.exists(sh):
            return None, f"choreo rc={p.returncode}: {(p.stderr or p.stdout)[-300:]}"
        subprocess.run(["bash", sh, "--compile-link"], capture_output=True,
                       text=True, cwd=REPO)
        e = subprocess.run(["stdbuf", "-o0", "-e0", "bash", sh, "--execute"],
                           capture_output=True, text=True, cwd=REPO)
        return e.returncode, e.stdout + e.stderr
    finally:
        shutil.rmtree(t, ignore_errors=True)


def _dump_raw(mid, label, rc, out):
    """Persist one arm's raw output so verdicts can be re-derived offline.

    The rc is stored in a sidecar line rather than parsed back out of the text,
    because `crash-only` is decided on rc and the text alone cannot recover it.
    """
    if not PROBE_RAW_DIR:
        return
    try:
        os.makedirs(PROBE_RAW_DIR, exist_ok=True)
        with open(os.path.join(PROBE_RAW_DIR, f"{mid}.{label}.txt"), "w") as f:
            f.write(f"# rc={rc}\n")
            f.write(out or "")
    except OSError:
        # Never let bookkeeping kill the probe.
        pass


# --------------------------------------------------------------------------
# mutated-access extraction
# --------------------------------------------------------------------------

def _norm_idx(expr):
    """Normalize an index expression so source and ledger spellings compare equal.

    choreo renders ledger index text with padding AND redundant parens:
    "` (q - 1) `" for the source's `q - 1`, "` ( (p # n)  + 1) `" for `p#n + 1`.
    The first cut compared whitespace-stripped strings only, so `q-1` never
    matched `(q-1)` and EVERY C3 candidate fell through to C4 ("not assessed")
    even though the ledger plainly carries the obligation for that index. That is
    how C3 came out as 0 on the second run after coming out as 4 on the first —
    the two runs disagreed because of a string-formatting detail, not because of
    anything choreo did.

    Stripping all parens is slightly aggressive (it would conflate `(a+b)*c` with
    `a+b*c`), but every index expression in this suite is a flat sum of loop
    variables and literals, and a false MATCH here is recoverable — the report
    prints the obligation text alongside, so a reader can see what was compared.
    A false MISS is not: it silently reclassifies a soundness bug as a coverage
    gap.
    """
    s = re.sub(r"\s+", "", expr or "")
    return s.replace("(", "").replace(")", "")


def _at_sites_in(text):
    """All `arr.at(...)` sites in a line, as {(arr, (idx, ...))}."""
    out = []
    for m in re.finditer(r"(\w+)\s*(?:\.data)?\.at\(([^)]*)\)", text):
        arr = m.group(1)
        idxs = tuple(a.strip() for a in m.group(2).split(","))
        out.append((arr, idxs))
    return out


def mutated_accesses(diff):
    """Parse the unified diff for the index positions the mutation ACTUALLY changed.

    Returns (at_sites, other_sites) where at_sites is [(array, [changed index
    expressions])] and other_sites is a sorted list of non-`.at` surface names
    (dma.copy chunk destinations, tile counts, output declarations, caller
    extents) — the surfaces choreo's assessor does not model (cause C4).

    WHY THIS IS DIFF-AWARE RATHER THAN `+`-LINE-ONLY. The first cut collected
    every `.at(...)` on a `+` line. A mutation changes one index inside a whole
    statement, so the `+` line also carries every UNCHANGED access on it: for
    `out_s.at(0, 0, q - 1, 0) = (inp_s.at(0, 0, q, 0) > 0.0f) ? ...` it reported
    `inp_s`'s `q` as mutated. That sent C3 chasing the wrong array and made the
    evidence in the report unreadable. So the `+` line's sites are matched
    positionally against the `-` line's, and only index positions that differ are
    reported. A site with no `-` counterpart (an added access) counts as fully
    mutated.
    """
    minus, plus, other = [], [], []
    for line in (diff or "").splitlines():
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-"):
            minus.append(line[1:])
        elif line.startswith("+"):
            plus.append(line[1:])

    at_sites, other_sites = [], []
    # Pair `-`/`+` lines by order. gen_mutants.py emits a contiguous hunk per
    # edit, so the i-th `+` line corresponds to the i-th `-` line.
    for i, pline in enumerate(plus):
        mline = minus[i] if i < len(minus) else ""
        before = _at_sites_in(mline)
        after = _at_sites_in(pline)
        for j, (arr, idxs) in enumerate(after):
            old = before[j][1] if j < len(before) and before[j][0] == arr else None
            if old is None:
                changed = list(idxs)          # added access: everything is new
            else:
                changed = [new for k, new in enumerate(idxs)
                           if k >= len(old) or old[k] != new]
            if changed:
                at_sites.append((arr, changed))
        # Non-`.at` surfaces. The pattern must capture the ARGUMENTS, not just the
        # callee: `inp.chunkat(` appears byte-identically on both the `-` and the
        # `+` line when the mutation is inside the argument list
        # (`chunkat(p#i, ...)` -> `chunkat(p#i + 1, ...)`), so comparing the
        # prefix alone classifies a real mutation as "unchanged" and loses the
        # surface entirely. group(1) is the name to report, group(0) is the whole
        # site used for the changed/unchanged test.
        for pat, name in (
                (r"=>\s*(\w+)\.chunkat\([^;]*", None),
                (r"(\w+)\.chunkat\([^;]*", None),
                (r"\bf32\s*\[[^\]]*\]\s*(\w+)\s*;", None),
                # `.span(N)` is an EXTENT query, not an element access: choreo's
                # assessor never models it, so a mutation here (a reduction divisor
                # `lhs.span(3)` -> `lhs.span(3) - 1`, or an output declared
                # `rhs.span(1) - 1`) has no obligation to violate. Without this
                # pattern such a diff yields NO surface at all and the mutant is
                # filed UNRESOLVED — "the parser found nothing" — which is a claim
                # about my parser, not about choreo. Naming the surface lets it be
                # attributed properly (C4 for a value error the assessor cannot
                # see, C5 when the description says it is out of scope).
                #
                # The `[^;]*` tail is load-bearing, and for the same reason as
                # `.chunkat`: the token `lhs.span(3)` appears BYTE-IDENTICALLY on
                # both diff lines, because what the mutation changes is the
                # arithmetic AROUND it (`/ lhs.span(3)` -> `/ (lhs.span(3) - 1)`).
                # Matching the bare accessor would classify a real mutation as
                # unchanged and lose the surface. The tail extends the match to the
                # end of the statement so the changed part is inside group(0).
                (r"(\w+)\.span\(\d+\)[^;]*", "span-extent"),
                (r"with index = \{[^}]*\} in \[([^\]]*)\]", "tile-count"),
                # `[^;]*` not `[^)]*`, for the nested-paren reason already given
                # twice above -- and here it is not hypothetical. The real call
                # sites in this suite nest:
                #   call k_matmul(l1_a.chunkat(i,_,_), l1_b.data, l1_out,
                #                 l1_a.span(1), l1_b.span(0), l1_a.span(2));
                # `[^)]*` stops at chunkat's OWN closing paren, so group(0) is
                # the truncated prefix `call k_matmul(l1_a.chunkat(i,_,_)`. That
                # prefix is byte-identical on the `-` and `+` lines whenever the
                # mutation is anywhere later in the argument list (the M3.s1.mm1
                # atom mutants mutate exactly those trailing `.span(N)` extents),
                # so the changed/unchanged test below says "unchanged", the
                # surface is dropped, and the mutant is filed UNRESOLVED -- a
                # claim about this parser rather than about choreo.
                (r"call \w+\([^;]*", "call-arg"),
                (r"foreach \w+ in \[([^\]]*)\]", "loop-bound"),
                (r"make_spandata<[^>]*>\([^;]*", "caller")):
            for m in re.finditer(pat, pline):
                if mline and m.group(0) in mline:
                    continue
                if name:
                    other_sites.append(name)
                else:
                    # The reported name is the captured array/extent token. For
                    # `call \w+(...)` there is no capture group, so fall back to
                    # the literal surface name.
                    other_sites.append(m.group(1) if m.groups() else "call-arg")
    return at_sites, sorted(set(other_sites))


def obligations_for(obligations, arr):
    """All ledger obligations mentioning element access '<arr>'."""
    out = []
    for o in obligations:
        m = RE_OBL.search(o.get("message", "").split(", /")[0])
        if m and m.group("arr") == arr:
            out.append((o, m))
    return out


def _is_relevant(m, at_sites, other_sites):
    """Does this ledger obligation concern a surface the mutation ACTUALLY touched?

    WHY THIS EXISTS. `layer_normalization/3_attention` carries 5 runtime / 4
    suppressed obligations whether or not it is mutated — they are the base
    kernel's own dynamic-shape guards on `lhs`/`out`'s `j` index (the same 5 that
    make the paper's 79/74/5/0 criterion come out right). A C1 test keyed only on
    "this mutant's ledger has suppressed obligations" therefore fires on EVERY
    mutant of that case, including ones whose mutation has nothing to do with `j`.
    Two did: `M2.s3.ln3.reduce` (changes a reduction divisor) and
    `M2.s4.ln3.caller.out` (changes a `make_spandata` extent in the host `main()`).
    Both were reported as "the assessor was RIGHT and the cost filter lost it" —
    and the execute probe contradicts both: with every guard forced on and hoisting
    disabled they are still caught only by the numeric oracle, because no bound
    guard can catch a wrong divisor or a wrong host-side buffer extent.

    So relevance is required. An obligation is relevant iff it names a mutated
    `.at` array AND the mutated index expression, or it names a mutated
    non-`.at` surface. The second arm is deliberately loose: a declaration or
    `dma.copy` destination is not index-specific, so any obligation on that array
    is plausibly a consequence of the mutation (for `M2.s4.ln1.out`, which
    declares `out` with extent `K - 1`, the relevant obligation is exactly
    "`out`'s 3rd index < 767"). Loose in this direction is safe because it can
    only keep a cause that the ledger already supports; the alternative — dropping
    it — is what produced the false C4s.
    """
    arr = m.group("arr")
    idx = _norm_idx(m.group("idx"))
    for a, idxs in at_sites:
        if a == arr and any(_norm_idx(i) == idx for i in idxs):
            return True
    return arr in other_sites


def relevant_runtime(obligations, at_sites, other_sites, enabled):
    """Runtime obligations that are `enabled` (or not) AND concern a mutated surface."""
    out = []
    for o in obligations:
        if o.get("outcome") != "runtime" or o.get("enabled") is not enabled:
            continue
        m = RE_OBL.search(o.get("message", "").split(", /")[0])
        if m and _is_relevant(m, at_sites, other_sites):
            out.append((o, m))
    return out


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def classify(rec, man, execute=False):
    """Attribute one `never` mutant to exactly one cause.

    Returns a dict with `cause`, the evidence, and (when relevant) the ledger
    rows that decided it. Causes are tested in the order C5, C1, C3, C2, C4 —
    C5 first because an out-of-scope value mutation must not be reported as a
    detection gap at all, and C4 last because it is the residual ("choreo said
    nothing about this access") and would otherwise swallow the others.
    """
    mid = rec["mutant_id"]
    src = mutant_path(mid)
    out = {
        "mutant_id": mid,
        "class": rec.get("class"),
        "spec": rec.get("spec"),
        "category": rec.get("category"),
        "case": rec.get("case"),
        "paper_category": rec.get("paper_category"),
        "desc": man.get("desc"),
        "detector_at_default": rec.get("detector"),
        "mutant_path": os.path.relpath(src, REPO) if src else None,
    }
    if not src:
        out.update(cause="UNRESOLVED", reason="mutant .co not found")
        return out

    at_sites, other_sites = mutated_accesses(man.get("diff"))
    out["mutated_at_sites"] = [f"{a}.at({', '.join(i)})" for a, i in at_sites]
    out["mutated_other_sites"] = other_sites

    obligations, stats_text = run_ledger(src)
    if obligations is None:
        out.update(cause="UNRESOLVED", reason=stats_text)
        return out

    by_outcome = collections.Counter(o.get("outcome") for o in obligations)
    out["ledger"] = {
        "n_obligations": len(obligations),
        "outcomes": dict(by_outcome),
        "runtime_enabled": sum(1 for o in obligations
                               if o.get("outcome") == "runtime" and o.get("enabled") is True),
        "runtime_suppressed": sum(1 for o in obligations
                                  if o.get("outcome") == "runtime" and o.get("enabled") is False),
    }
    m = re.search(r"(\d+)\s+assess\s+-\s+Runtime assertions disabled by cost filter",
                  stats_text or "")
    if m:
        out["ledger"]["stats_disabled_by_cost_filter"] = int(m.group(1))

    # ---- C5: out of scope (value change, no bound violation) ----
    # specs §4 puts numeric correctness out of scope, so a mutation that cannot
    # move an index out of bounds has no obligation to violate and `never` is
    # the CORRECT verdict for it — not a detection gap.
    #
    # The test is on the CHANGED index expressions only (mutated_accesses already
    # differenced the `-`/`+` lines). Three shapes qualify:
    #   (a) an index collapsed to a constant          `k`  -> `0`
    #   (b) an index replaced by another in-range var  `q + K` -> `q`
    #   (c) a loop bound shortened                     `[K]` -> `[K - 1]`
    # None of them can produce an out-of-range index. An OFFSET (`k + 1`,
    # `q - 1`, `p#n + 1`) can, and is never C5.
    #
    # The ledger must agree (no runtime obligation at all) and the mutant's own
    # description must be value-vocabulary, so this cannot silently absorb a real
    # bound violation whose diff happens to look benign.
    changed_idx = [i for _, idxs in at_sites for i in idxs]
    benign_index = all(
        re.fullmatch(r"-?\d+(\.\d+f?)?|[A-Za-z_]\w*", i) for i in changed_idx)
    # `span-extent` qualifies because a `.span(N)` query returns an EXTENT, not an
    # index: `lhs.span(3)` -> `lhs.span(3) - 1` makes a reduction divide by the
    # wrong number, which is arithmetic, not an out-of-range access.
    benign_surface = set(other_sites) <= {"loop-bound", "span-extent"}

    # Relevance-filtered runtime obligations. The RAW counts are kept in
    # out["ledger"] for auditability, but the CAUSE tests use these: a base
    # kernel's own dynamic-shape guards are present in every mutant of that case
    # and say nothing about the mutation. See _is_relevant.
    rel_suppressed = relevant_runtime(obligations, at_sites, other_sites, False)
    rel_enabled = relevant_runtime(obligations, at_sites, other_sites, True)
    out["ledger"]["relevant_runtime_suppressed"] = len(rel_suppressed)
    out["ledger"]["relevant_runtime_enabled"] = len(rel_enabled)

    if (benign_index and benign_surface
            and (changed_idx or other_sites)
            and not rel_suppressed
            and not rel_enabled
            and re.search(r"zero-stride|collapse|duplicate|partial|overwrite|"
                          r"omitted tail|shortened|divisor|reduction",
                          (man.get("desc") or ""), re.I)):
        out.update(cause="C5_OUT_OF_SCOPE",
                   reason="the mutation cannot move an index out of bounds "
                          "(constant collapse / in-range variable substitution / "
                          "shortened loop / wrong extent query): a VALUE error, "
                          "which specs §4 puts out of scope. `never` is the "
                          "correct verdict.",
                   changed_indices=changed_idx,
                   changed_surfaces=other_sites)
        return out
    # No mutated surface at all. This is a statement about the DIFF PARSER, not
    # about choreo, so it must not be allowed to masquerade as a coverage gap
    # (C4) further down. It sits before C1 as well: C1 is now relevance-gated, so
    # with no surface there is nothing for a suppressed obligation to be relevant
    # TO and C1 could not fire anyway.
    if not at_sites and not other_sites:
        out.update(cause="UNRESOLVED",
                   reason="the diff parser found no mutated access surface; the "
                          "diff needs manual reading")
        return out

    # ---- C1: cost-filter suppression ----
    if rel_suppressed:
        # Obligations that ARE suppressed but concern some OTHER access — almost
        # always the base kernel's own dynamic-shape guards. Recorded so the
        # report can show that C1 was decided on the mutation's obligation and not
        # on an inherited one.
        irrelevant = [o for o in obligations
                      if o.get("outcome") == "runtime" and o.get("enabled") is False]
        out.update(
            cause="C1_COST_FILTER_SUPPRESSED",
            reason="the assessor produced a `runtime` obligation for the mutated "
                   "access, but its cost tier exceeds the default -rtc level "
                   "(measured: default == entry == low), so the cost filter "
                   "dropped the guard. An EMISSION POLICY, not a soundness gap.",
            suppressed_obligations=[
                {"message": o["message"].split(", /")[0],
                 "cost": o.get("cost"), "mechanism": o.get("mechanism"),
                 "usage": o.get("usage")} for o, _ in rel_suppressed],
            suppressed_but_not_about_this_mutation=[
                o["message"].split(", /")[0] for o in irrelevant
                if o not in [x for x, _ in rel_suppressed]],
        )
        # C2 may ALSO apply (the guard, if forced on, may be hoisted somewhere
        # vacuous). Test it so the report can say whether forcing -rtc=all would
        # actually have caught this one.
        if execute:
            out["forced_all"] = probe_execute(src, mid)
        else:
            out["forced_all"] = {"status": "untested",
                                 "reason": "pass --execute to test C2 on this mutant"}
        return out

    # ---- C3 / C4: was the mutated access assessed at all? ----
    assessed = []
    for arr, idxs in at_sites:
        obs = obligations_for(obligations, arr)
        if not obs:
            continue
        for idx in idxs:
            # Match the mutated index expression against the obligation's index
            # text. choreo renders it with padding AND redundant parens, e.g.
            # "` (k + 1) `" for the source's `k + 1`, so both sides go through
            # _norm_idx. Comparing whitespace-stripped strings only was the bug
            # that made C3 unreachable for every expression index.
            norm = _norm_idx(idx)
            hits = [(o, m) for o, m in obs if _norm_idx(m.group("idx")) == norm]
            if not hits:
                continue
            bounds = {("upper" if UPPER in m.group("bound") else
                       "lower" if LOWER in m.group("bound") else "other")
                      for o, m in hits}
            assessed.append({
                "array": arr, "index": idx,
                "bounds_generated": sorted(bounds),
                "outcomes": sorted({o.get("outcome") for o, m in hits}),
                "n_obligations": len(hits),
            })
    out["assessed_mutated_indices"] = assessed

    if assessed:
        # The access WAS assessed. If only one bound was generated and the
        # mutation moves the index in the direction of the missing bound, the
        # obligation set is incomplete.
        incomplete = [a for a in assessed
                      if len(a["bounds_generated"]) == 1]
        if incomplete and not rel_enabled:
            out.update(
                cause="C3_LOWER_BOUND_OMITTED",
                reason="choreo assessed the mutated index but generated only ONE "
                       "of its two bounds. Every literal index in the same kernel "
                       "gets both (`>= 0` and `< N`); the mutated expression gets "
                       "one. When the mutation moves the index in the direction of "
                       "the missing bound (q-1 at q==0) nothing can fire. A "
                       "croqtile obligation-generation bug.",
                incomplete_indices=incomplete,
            )
            if execute:
                out["forced_all"] = probe_execute(src, mid)
            return out

        # Both bounds generated AND a guard actually emitted (a relevant
        # `enabled` runtime obligation), yet E1 recorded no detection. That is not
        # a coverage gap — choreo assessed the access, generated the obligation,
        # and emitted the check. The remaining explanation is PLACEMENT: the
        # assertion-hoisting pass can move the guard out of the loop body to a
        # point after the induction variable has been reset (`__iv_k = 0;`), where
        # it evaluates `0 + 1 < 768` — vacuously true. This is cause C2.
        #
        # Labelling it C4 ("no obligation generated") was flatly wrong and would
        # have understated choreo: C4 is the honest "we never looked at this
        # access", C2 is "we looked, we emitted, we put it where it cannot fire".
        # Only `--execute` can confirm C2, because the confirmation is that
        # `--disable-assert-hoist` makes the SAME guard text fire. Without it the
        # cause is reported as a C2 CANDIDATE, not as a confirmed defect.
        if rel_enabled:
            out.update(
                cause="C2_HOISTING_CANDIDATE",
                reason="both bounds were generated and a guard was EMITTED for the "
                       "mutated access, yet nothing fired at the default -rtc "
                       "level. The remaining explanation is placement: the "
                       "assertion-hoisting pass can move the guard past the "
                       "induction variable's reset, where it is vacuously true. "
                       "Confirmed only by --execute (same guard text fires under "
                       "--disable-assert-hoist).",
                emitted_obligations=[
                    {"message": o["message"].split(", /")[0],
                     "cost": o.get("cost")} for o, _ in rel_enabled],
                incomplete_indices=incomplete,
            )
            out["forced_all"] = (probe_execute(src, mid) if execute else
                                 {"status": "untested",
                                  "reason": "pass --execute to confirm the hoisting "
                                            "defect on this mutant"})
            return out

        out.update(cause="C4_NOT_ASSESSED",
                   reason="the mutated index was assessed and both bounds "
                          "generated, yet no guard was emitted and none fired — "
                          "needs manual inspection",
                   )
        return out

    # Nothing matched: either the array is absent from the ledger entirely, or
    # only non-`.at` surfaces were mutated (dma.copy destinations, tile counts,
    # output declarations, caller extents).
    arrays_in_ledger = sorted({
        RE_OBL.search(o.get("message", "").split(", /")[0]).group("arr")
        for o in obligations
        if RE_OBL.search(o.get("message", "").split(", /")[0])})
    out["arrays_assessed_by_choreo"] = arrays_in_ledger
    wanted = sorted({a for a, _ in at_sites} | set(other_sites))
    out["arrays_mutated"] = wanted
    missing = [a for a in wanted if a not in arrays_in_ledger]
    out.update(
        cause="C4_NOT_ASSESSED",
        reason="choreo generated NO obligation for the mutated access, so there "
               "was nothing to suppress and nothing to hoist. A coverage gap in "
               "the assessor's model of these surfaces (dma.copy chunk "
               "destinations, `with index` tile counts, output declarations, "
               "caller-side extents) — not a lost verdict.",
        arrays_with_no_obligation=missing,
    )
    if execute:
        out["forced_all"] = probe_execute(src, mid)
    return out


def probe_execute(src, mid=""):
    """Run the mutant twice at `-rtc=all`: hoisting ON (default) vs OFF.

    This is the C2 test. If choreo's own index guard fires only in the
    hoisting-OFF arm, the guard was emitted but placed where it cannot fire —
    the assertion-hoisting pass moved it past the induction variable's reset.

    Verdict vocabulary (see the three-source note above RE_CHOREO_GUARD):
      choreo-guard-fired   source A, device form (stdout, location suffix)
                                                            -> DETECTION
      entry-check-fired    source A, host form (stderr, location suffix)
                                                            -> DETECTION
      shape-check-fired    source A, host form, `shape inconsistent` template
                           (stderr, no location suffix)     -> DETECTION
      runtime-lib-bounds   source B ArrayProxy bounds check -> NOT a detection
      lib-sanity-check     ungated runtime/choreo.h sanity check (`at::Tensor`,
                           `[choreo-rt]`)                   -> NOT a detection
      oracle-only          source C §7 reference check      -> NOT a detection
      unattributed-assert  an assertion matched no vocabulary -> needs a human
      unattributed-rtc     a runtime-check line matched no vocabulary
                                                            -> needs a human
      crash-only           aborted with no message (device trap)
      passed-no-detection  ran to completion, nothing fired

    `runtime-lib-bounds`, `lib-sanity-check` and the two `unattributed-*`
    verdicts are kept out of `oracle-only` on purpose. Folding B into the oracle
    would report a choreo-provided bounds check as the harness's work; folding
    an unrecognised message into the oracle would silently credit the harness
    with anything the vocabulary missed. All are non-detections for S2 either
    way, so the split costs nothing and preserves the audit trail.

    `mid` is used only to name the persisted raw log; it does not affect the
    verdict.
    """
    res = {}
    for label, extra in (("hoist_on", []), ("hoist_off", ["--disable-assert-hoist"])):
        rc, out = run_execute(src, extra)
        if mid:
            _dump_raw(mid, label, rc, out)
        if rc is None:
            res[label] = {"status": "compile-fail", "detail": out}
            continue
        # Source A first, and positively. It must precede the bare
        # "runtime check failed" prefix because A-host and the ungated library
        # sanity checks share that prefix.
        if RE_CHOREO_GUARD.search(out):
            verdict = "choreo-guard-fired"
            detail = RE_CHOREO_GUARD.search(out).group(0)[:160]
        elif RE_ENTRY_CHECK.search(out):
            verdict = "entry-check-fired"
            detail = RE_ENTRY_CHECK.search(out).group(0)[:160]
        elif RE_SHAPE_CHECK.search(out):
            verdict = "shape-check-fired"
            detail = RE_SHAPE_CHECK.search(out).group(0)[:160]
        elif RE_LIB_SANITY.search(out):
            verdict = "lib-sanity-check"
            detail = RE_LIB_SANITY.search(out).group(0)[:160]
        elif RE_ANY_RTC.search(out):
            # A runtime-check line that is neither a known assessment nor a
            # known library check. Do NOT assume assessment: that inference is
            # what would credit an allocation failure to the assessor.
            verdict = "unattributed-rtc"
            detail = RE_ANY_RTC.search(out).group(0)[:160]
        elif RE_RUNTIME_LIB.search(out):
            verdict = "runtime-lib-bounds"
            detail = RE_RUNTIME_LIB.search(out).group(0)[:160]
        elif RE_ORACLE.search(out):
            verdict = "oracle-only"
            detail = RE_ORACLE.search(out).group(0)[:160]
        elif RE_ANY_ASSERT.search(out):
            verdict = "unattributed-assert"
            detail = RE_ANY_ASSERT.search(out).group(0)[:160]
        elif rc != 0:
            verdict = "crash-only"
            detail = f"rc={rc}"
        else:
            verdict = "passed-no-detection"
            detail = ""
        res[label] = {"status": verdict, "rc": rc, "detail": detail}
    on, off = res.get("hoist_on", {}), res.get("hoist_off", {})
    res["hoisting_defect_confirmed"] = (
        off.get("status") in DETECTION_VERDICTS
        and on.get("status") not in DETECTION_VERDICTS)
    return res


# --------------------------------------------------------------------------

# C2 appears under two labels because it is the only cause that CANNOT be decided
# from the ledger alone. `C2_HOISTING_CANDIDATE` is what the ledger-only pass
# reports when the obligation was generated AND a guard was emitted, so placement
# is the only remaining explanation; `HOISTING_DEFECT` is what `--execute` promotes
# it to once `--disable-assert-hoist` makes the same guard text fire. Keeping them
# distinct means a reader of the ledger-only artifact is never told a codegen bug
# is confirmed when it is only inferred.
CAUSE_ORDER = ["C1_COST_FILTER_SUPPRESSED", "C2_HOISTING_CANDIDATE",
               "HOISTING_DEFECT", "C3_LOWER_BOUND_OMITTED",
               "C4_NOT_ASSESSED", "C5_OUT_OF_SCOPE", "UNRESOLVED"]

CAUSE_MEANING = {
    "C1_COST_FILTER_SUPPRESSED":
        "assessor was RIGHT; the cost filter suppressed the guard (emission policy)",
    "C2_HOISTING_CANDIDATE":
        "guard EMITTED for the mutated index yet nothing fired — placement "
        "suspected, needs --execute to confirm (croqtile codegen bug)",
    "HOISTING_DEFECT":
        "guard emitted but hoisted past the induction variable's reset — VACUOUS, "
        "CONFIRMED by --disable-assert-hoist (croqtile codegen bug)",
    "C3_LOWER_BOUND_OMITTED":
        "only one of the index's two bounds was generated (croqtile soundness bug)",
    "C4_NOT_ASSESSED":
        "no obligation generated for the mutated access (assessor coverage gap)",
    "C5_OUT_OF_SCOPE":
        "value error, not a bound violation — `never` is the CORRECT verdict",
    "UNRESOLVED":
        "could not be attributed; inspect manually",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", default=os.path.join(RAW, "e1_mutant_records.json"))
    ap.add_argument("--manifest", default=os.path.join(RAW, "mutant_manifest.json"))
    ap.add_argument("--out", default=os.path.join(RAW, "never_attribution.json"))
    ap.add_argument("--execute", action="store_true",
                    help="also run the C2 probe (nvcc + GPU, ~20 s per mutant). "
                         "Without it, C2 is reported as untested rather than "
                         "folded into another cause.")
    ap.add_argument("--only", default="", help="comma-separated mutant_ids")
    ap.add_argument("--raw-logs", default=os.path.join(RAW, "probe_raw"),
                    help="directory for each probe arm's raw stdout+stderr, "
                         "named <mutant_id>.<arm>.txt. Default raw/probe_raw; "
                         "pass an empty string to disable. See PROBE_RAW_DIR for "
                         "why this exists: it is what makes a later classifier "
                         "correction a re-projection over stored evidence "
                         "instead of a fresh nvcc run per mutant.")
    ap.add_argument("--include-noop", action="store_true",
                    help="attribute `noop` mutants too (default: only "
                         "`manifest=corrupts`, the admissible population)")
    a = ap.parse_args()

    if not os.path.exists(CHOREO):
        print(f"[choreo] no binary at {CHOREO}", file=sys.stderr)
        return 2

    # Module-level so probe_execute() can reach it without threading a parameter
    # through classify()'s four call sites.
    global PROBE_RAW_DIR
    PROBE_RAW_DIR = os.path.abspath(a.raw_logs) if a.raw_logs else None
    if PROBE_RAW_DIR and a.execute:
        os.makedirs(PROBE_RAW_DIR, exist_ok=True)

    with open(a.records) as f:
        data = json.load(f)
    rows = data.get("records", data if isinstance(data, list) else [])
    with open(a.manifest) as f:
        man_all = json.load(f)
    man_rows = man_all.get("mutants", man_all if isinstance(man_all, list) else [])
    byid = {m["mutant_id"]: m for m in man_rows}

    only = {x.strip() for x in a.only.split(",") if x.strip()}
    targets = [r for r in rows
               if r.get("outcome") == "never"
               and (a.include_noop or r.get("manifest") == "corrupts")
               and (not only or r.get("mutant_id") in only)]

    ver = data.get("toolchain_version") or rows[0].get("toolchain_version")
    print(f"[choreo] never-attribution: {len(targets)} mutant(s), "
          f"toolchain {str(ver)[:12]}, execute={a.execute}", file=sys.stderr)

    results = []
    for i, r in enumerate(targets, 1):
        mid = r["mutant_id"]
        print(f"[choreo] [{i:>3}/{len(targets)}] {mid}", file=sys.stderr)
        res = classify(r, byid.get(mid, {}), execute=a.execute)
        # Promote the C2 verdict into the cause when the probe confirms it: the
        # hoisting defect is the more specific and more actionable finding, and
        # leaving it nested inside C1 would bury a codegen bug under a policy note.
        fa = res.get("forced_all") or {}
        if fa.get("hoisting_defect_confirmed"):
            res["cause_under_execute"] = res["cause"]
            res["cause"] = "HOISTING_DEFECT"
        results.append(res)

    counts = collections.Counter(r["cause"] for r in results)
    payload = {
        "toolchain": "choreo",
        "toolchain_version": ver,
        "produced_by": "analyze_never.py",
        "n_never_analysed": len(results),
        "population": ("all `never` mutants" if a.include_noop
                       else "`manifest=corrupts` only (the admissible population; "
                            "`noop` mutants are discarded per specs §11.1)"),
        "execute_probe": a.execute,
        # Only meaningful when the probe actually ran. Reporting the directory
        # on a ledger-only run implies raw evidence backs this artifact, when in
        # fact nothing was written there -- and a reader who later corrects a
        # matcher would try (and fail) to re-project from it.
        "probe_raw_dir": (os.path.relpath(PROBE_RAW_DIR, REPO)
                          if (a.execute and PROBE_RAW_DIR) else None),
        "cause_counts": dict(counts),
        "cause_meaning": CAUSE_MEANING,
        "results": results,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\n[choreo] wrote {os.path.relpath(a.out, REPO)}", file=sys.stderr)

    report(payload)
    return 0


def report(p):
    line = "=" * 78
    print(f"\n{line}\nWHY DID {p['n_never_analysed']} MUTANTS COME BACK "
          f"`never`?\n{line}")
    print(f"  population: {p['population']}")
    print(f"  C2 execute probe: {'RUN' if p['execute_probe'] else 'NOT RUN'}")
    print(f"\n  {'cause':<30}{'n':>4}   meaning")
    for c in CAUSE_ORDER:
        n = p["cause_counts"].get(c, 0)
        if not n:
            continue
        print(f"  {c:<30}{n:>4}   {CAUSE_MEANING[c]}")
    for c, n in sorted(p["cause_counts"].items()):
        if c not in CAUSE_ORDER:
            print(f"  {c:<30}{n:>4}")

    det = sum(p["cause_counts"].get(c, 0) for c in
              ("C1_COST_FILTER_SUPPRESSED", "C2_HOISTING_CANDIDATE",
               "HOISTING_DEFECT", "C3_LOWER_BOUND_OMITTED"))
    gap = p["cause_counts"].get("C4_NOT_ASSESSED", 0)
    oos = p["cause_counts"].get("C5_OUT_OF_SCOPE", 0)
    tot = p["n_never_analysed"]
    print(f"\n{line}\nTHE SENTENCE THIS SUPPORTS\n{line}")
    print(f"  Of {tot} admissible `never` mutants:")
    print(f"    {det:>3}  choreo's assessor IDENTIFIED the violation and lost it "
          f"downstream")
    print(f"          (cost filter / hoisting placement / omitted bound)")
    print(f"    {gap:>3}  choreo generated NO obligation for the mutated access "
          f"(coverage gap)")
    print(f"    {oos:>3}  the mutation is a VALUE error, out of scope per specs §4 "
          f"— `never` is correct")
    if det + gap + oos != tot:
        print(f"    {tot - det - gap - oos:>3}  unattributed — inspect manually")
    print()
    print("  So `never` is NOT a single number. Reporting it as 'choreo missed "
          f"{tot} bugs'")
    print("  would be wrong for the first group (the verdict was reached, then "
          "dropped),")
    print("  wrong for the third (there was no bug of the kind E1 measures), and "
          "right only")
    print("  for the second. The coordinator decides how S2 is worded; this file "
          "is the")
    print("  evidence either way, and every row carries the ledger rows that "
          "decided it.")

    print(f"\n{line}\nPER-MUTANT DETAIL\n{line}")
    for r in p["results"]:
        print(f"  {r['mutant_id']:<26} {r['cause']}")
        print(f"      {r['class']}.s{r['spec']}/{r['category']}: {r['desc']}")
        if r.get("mutated_at_sites"):
            print(f"      mutated: {', '.join(r['mutated_at_sites'][:3])}")
        if r.get("mutated_other_sites"):
            print(f"      surfaces: {', '.join(r['mutated_other_sites'][:4])}")
        led = r.get("ledger") or {}
        if led:
            print(f"      ledger: {led.get('n_obligations')} obligations, "
                  f"outcomes={led.get('outcomes')}, "
                  f"suppressed={led.get('runtime_suppressed')}, "
                  f"enabled={led.get('runtime_enabled')}")
        for o in (r.get("suppressed_obligations") or [])[:2]:
            print(f"        suppressed: [{o.get('cost')}] {o['message'][:100]}")
        for a_ in (r.get("incomplete_indices") or [])[:3]:
            print(f"        bounds generated for `{a_['index']}` on "
                  f"'{a_['array']}': {a_['bounds_generated']} "
                  f"(outcomes={a_['outcomes']})")
        if r.get("arrays_with_no_obligation"):
            print(f"      NO obligation for: "
                  f"{', '.join(r['arrays_with_no_obligation'][:4])}")
        fa = r.get("forced_all") or {}
        if fa.get("status") == "untested":
            print(f"      C2 probe: untested (pass --execute)")
        elif fa:
            on, off = fa.get("hoist_on", {}), fa.get("hoist_off", {})
            print(f"      C2 probe @-rtc=all: hoist_on={on.get('status')}  "
                  f"hoist_off={off.get('status')}  "
                  f"defect={'CONFIRMED' if fa.get('hoisting_defect_confirmed') else 'no'}")
        print()


if __name__ == "__main__":
    sys.exit(main())
